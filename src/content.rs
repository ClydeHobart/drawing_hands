use {
    bitvec::{order::Lsb0, vec::BitVec, view::AsBits},
    clap::{ArgAction::HelpLong, Parser},
    glam::UVec2,
    image::{ImageError, Pixel, Rgb, Rgba, RgbaImage},
    nom::{
        branch::alt,
        bytes::complete::{tag, take_while, take_while1},
        character::complete::{digit0, line_ending, multispace0, not_line_ending, one_of, satisfy},
        combinator::{all_consuming, iterator, map, map_parser, map_res, success},
        error::Error as NomError,
        sequence::{preceded, terminated, tuple},
        IResult,
    },
    std::{
        alloc::{alloc_zeroed, handle_alloc_error, Layout},
        collections::{HashSet, VecDeque},
        ffi::OsStr,
        fmt::{Debug, Formatter, Result as FmtResult},
        fs::{create_dir_all, read_to_string, File},
        hash::{DefaultHasher, Hash, Hasher},
        io::{Error as IoError, ErrorKind, Read, Write},
        iter::Iterator,
        mem::MaybeUninit,
        ops::Range,
        path::Path,
        process::{Child, ChildStdin, ChildStdout, Command, Stdio},
        ptr::NonNull,
        str::FromStr,
        sync::{
            atomic::{AtomicU64, Ordering},
            Arc, Mutex,
        },
        thread::{sleep, spawn},
        time::{Duration, Instant},
    },
};

macro_rules! output_file { ($($file:expr)?) => { concat!("./output" $(, "/", $file)?) }; }

const TAB_SIZE: usize = 4_usize;
const IMAGE_TEXT_PATH: &str = "";
const OUTPUT_IMAGE_PATH: &str = output_file!("");

/* This consists of `MANIFEST_BIT_LEN`* blocks of `BLOCK_BYTE_LEN` payload bytes. Before the */
/* initial block, and between every `FRAME_BLOCK_LEN` blocks, is `NEW_LINE_SLICE`. Each block is */
/* either completely parity payload, or completely drawing payload. The drawing is centered in */
/* the payload. If the drawing is landscape, it is 384 bytes wide and 128 bytes tall. If the */
/* drawing is portrait, it is 256 bytes wide and 192 bytes tall. Following the payload bytes are */
/* `MANIFEST_BYTE_LEN` manifest bytes. Each manifest byte encodes 6 bits of info (in base-64) of */
/* describing whether the next 6 blocks are drawing or parity. */
/* *Assume constants are associated constants of `CharAlphaGridArr`. */
const BYTES: &[u8] = br"";

unsafe fn alloc_zeroed_non_null<T>() -> NonNull<T> {
    let layout: Layout = Layout::new::<T>();

    if let Some(non_null) = NonNull::new(alloc_zeroed(layout) as *mut T) {
        non_null
    } else {
        handle_alloc_error(layout)
    }
}

pub fn alloc_zeroed_box<T>() -> Box<T> {
    /* SAFETY: The calls to `alloc_zeroed_non_null` and `Box::from_raw` are safe because the */
    /* value returned by the former has ownership immediately assumed by the latter: one */
    /* allocation, one deallocation (when the value returned by this function is dropped). */
    unsafe { Box::from_raw(alloc_zeroed_non_null().as_ptr()) }
}

fn parse_line(input: &str) -> IResult<&str, &str> {
    terminated(not_line_ending, line_ending)(input)
}

fn parse_whitespace_then_non_whitespace(input: &str) -> IResult<&str, (&str, &str)> {
    tuple((multispace0, take_while1(|c: char| !c.is_ascii_whitespace())))(input)
}

#[derive(Clone, Copy)]
struct Pos(UVec2);

impl Debug for Pos {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_fmt(format_args!("({},{})", self.0.x, self.0.y))
    }
}

#[derive(Clone, Debug)]
struct TextPos<'s> {
    text: &'s str,
    pos: Pos,
}

fn iter_text_poses(file_str: &str, tab_size: usize) -> impl Iterator<Item = TextPos> + '_ {
    let tab_size: u32 = tab_size as u32;
    let mut y: u32 = 0_u32;

    iterator(file_str, parse_line)
        .collect::<Vec<&str>>()
        .into_iter()
        .map(move |line| {
            let line_y: u32 = y;

            y += 1;

            (line, line_y)
        })
        .flat_map(move |(line, y)| {
            let mut x: u32 = 0;

            iterator(line, parse_whitespace_then_non_whitespace)
                .collect::<Vec<(&str, &str)>>()
                .into_iter()
                .map(move |(whitespace, text)| {
                    for c in whitespace.chars() {
                        x += if c == '\t' {
                            tab_size - x % tab_size
                        } else {
                            1
                        };
                    }

                    let pos: Pos = Pos(UVec2::new(x, y));

                    x += text.len() as u32;

                    TextPos { text, pos }
                })
        })
}

fn try_get_vim_string(file_path: &str) -> Result<String, String> {
    const INITIAL_VIEW_BUFFER_SLEEP: Duration = Duration::from_millis(500_u64);
    const INTERRUPTED_SLEEP: Duration = Duration::from_millis(1_u64);
    const MAX_WATCHER_UPDATE_TIME: Duration = Duration::from_millis(50_u64);
    const WATCHER_SLEEP: Duration = Duration::from_millis(100_u64);
    const KILL_CHILD_SLEEP: Duration = Duration::from_millis(500_u64);

    let vim_string_file_path: Option<String> = Path::file_stem(file_path.as_ref())
        .and_then(OsStr::to_str)
        .map(|file_stem| format!("./output/{file_stem}.vim.txt"));

    if let Some(vim_string_file_path) = vim_string_file_path.as_ref() {
        if Path::is_file(vim_string_file_path.as_ref()) {
            if let Ok(vim_string) = read_to_string(vim_string_file_path) {
                return Ok(vim_string);
            }
        }
    }

    let mut child: Child = Command::new("vim")
        .args([
            "--cmd",
            "let g:loaded_matchparen=1",
            "-c",
            "se noru bo=insertmode,esc,cursor ttyfast ttymouse=xterm",
            file_path,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| format!("{e}"))?;

    let mut stdin: ChildStdin = child
        .stdin
        .take()
        .ok_or_else(|| "No stdin available for child process.".to_string())?;

    let mut stdout: ChildStdout = child
        .stdout
        .take()
        .ok_or_else(|| "No stdout available for child process.".to_string())?;

    let byte_count: usize = File::open(file_path)
        .ok()
        .map(|file| file.bytes().filter_map(Result::ok).count())
        .unwrap_or_default();

    let full_block_read_len: usize =
        (1_usize << (usize::BITS - byte_count.leading_zeros()).saturating_sub(1_u32)).max(2_usize);

    /* Sleep so that the vim can render the initial view buffer. */
    sleep(INITIAL_VIEW_BUFFER_SLEEP);

    let start: Instant = Instant::now();

    let now_nanos = move || (Instant::now() - start).as_nanos() as u64;

    struct ThreadData {
        stdout_bytes: Mutex<Vec<u8>>,
        update_time: AtomicU64,
    }

    /* This capacity is an underestimate, but it'll save allocations for a few orders of */
    /* magnitude. */
    let thread_data: Arc<ThreadData> = Arc::new(ThreadData {
        stdout_bytes: Mutex::new(Vec::with_capacity(byte_count)),
        update_time: AtomicU64::new(0_u64),
    });

    let worker_thread_data = thread_data.clone();

    let worker = spawn(move || {
        let mut buffer: Vec<u8> = vec![0_u8; full_block_read_len];
        let mut total: usize = 0_usize;

        loop {
            match stdout.read(&mut buffer[..]) {
                Ok(0_usize) => {
                    println!("read 0, total {total}");

                    break;
                }
                Ok(read_len) => {
                    total += read_len;

                    println!("read {read_len}, total {total}");

                    if read_len == 1_usize && buffer[0_usize] == b'\x07' {
                        break;
                    }

                    let mut stdout = worker_thread_data.stdout_bytes.lock().expect(
                        "While worker is running, it should be the only thread accessing its thread data.",
                    );
                    stdout.extend_from_slice(&buffer[..read_len]);
                }
                Err(error) if error.kind() == ErrorKind::Interrupted => {
                    sleep(INTERRUPTED_SLEEP);
                }
                Err(_) => {
                    break;
                }
            }

            worker_thread_data
                .update_time
                .store(now_nanos(), Ordering::Release);
        }
    });

    let mut loop_watcher = |send_down: bool| loop {
        if worker.is_finished() {
            break;
        }

        let update_time: u64 = thread_data.update_time.load(Ordering::Acquire);

        if update_time != 0_u64
            && Duration::from_nanos(now_nanos() - update_time) > MAX_WATCHER_UPDATE_TIME
        {
            if !send_down {
                break;
            }

            stdin.write_all("\x1B[B".as_bytes()).ok();
        }

        sleep(WATCHER_SLEEP);
    };

    /* First loop the watcher until the worker reaches the end of the initial display. */
    loop_watcher(false);

    /* Then loop the watcher pressing down. The worker should receive a bell eventually, at which */
    /* point the worker will be finished and the watcher will end its loop. */
    loop_watcher(true);

    /* Kill the child process to keep that from staying open. */
    child.kill().ok();

    /* Sleep in case killing the child process now finishes the worker. */
    sleep(KILL_CHILD_SLEEP);

    if !worker.is_finished() {
        Err("Worker wasn't finished when it needs to be".to_string())
    } else {
        let stdout = std::mem::take(
            &mut *thread_data
                .stdout_bytes
                .lock()
                .expect("Worker was finished, but someone else had the thread data locked."),
        );

        let vim_string: Result<String, String> =
            String::from_utf8(stdout).map_err(|e| format!("{e}"));

        if let (Some(vim_string_file_path), Ok(vim_string)) =
            (vim_string_file_path, vim_string.as_ref())
        {
            if let Ok(mut file) = File::create(vim_string_file_path) {
                file.write_all(vim_string.as_bytes()).ok();
            }
        }

        vim_string
    }
}

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Copy, PartialEq)]
enum ControlSequence {
    SgrReset,
    SgrSetForeground(Rgb<u8>),
    SgrSetBackground(Rgb<u8>),
    CursorPosition(UVec2),
    Other,
}

impl ControlSequence {
    fn apply(&self, terminal_colors: &mut TerminalColors) {
        match self {
            Self::SgrReset => *terminal_colors = TerminalColors::default(),
            Self::SgrSetForeground(fg) => terminal_colors.foreground = *fg,
            Self::SgrSetBackground(bg) => terminal_colors.background = *bg,
            Self::CursorPosition(_) => (),
            Self::Other => (),
        };
    }

    fn proceeds_text(&self) -> bool {
        matches!(
            self,
            Self::SgrSetForeground(_) | Self::SgrSetBackground(_) | Self::CursorPosition(_)
        )
    }
}

fn parse_u32(input: &str) -> IResult<&str, u32> {
    map_res(digit0, |digits: &str| {
        if digits.is_empty() {
            Ok(0_u32)
        } else {
            u32::from_str(digits)
        }
    })(input)
}

fn is_control_code(c: char) -> bool {
    matches!(c, '\u{7}' | '\u{8}' | '\u{c}' | '\u{1b}')
}

fn parse_non_control_code(input: &str) -> IResult<&str, &str> {
    take_while(|c: char| !is_control_code(c))(input)
}

fn parse_control_sequence(input: &str) -> IResult<&str, ControlSequence> {
    alt((
        preceded(
            tag("\u{1b}["),
            alt((
                terminated(
                    map_parser(
                        take_while(|c: char| !c.is_ascii_alphabetic()),
                        alt((
                            map(all_consuming(success(())), |_| ControlSequence::SgrReset),
                            map(
                                tuple((
                                    one_of("34"),
                                    tag("8;2;"),
                                    parse_u32,
                                    tag(";"),
                                    parse_u32,
                                    tag(";"),
                                    parse_u32,
                                )),
                                |(c, _, r, _, g, _, b)| {
                                    let rgb: Rgb<u8> = Rgb([r as u8, g as u8, b as u8]);

                                    if c == '3' {
                                        ControlSequence::SgrSetForeground(rgb)
                                    } else {
                                        ControlSequence::SgrSetBackground(rgb)
                                    }
                                },
                            ),
                        )),
                    ),
                    tag("m"),
                ),
                map(
                    tuple((parse_u32, tag(";"), parse_u32, tag("H"))),
                    |(x, _, y, _)| ControlSequence::CursorPosition(UVec2::new(x, y)),
                ),
            )),
        ),
        map(
            tuple((satisfy(is_control_code), parse_non_control_code)),
            |_| ControlSequence::Other,
        ),
    ))(input)
}

fn parse_control_sequence_then_text(input: &str) -> IResult<&str, (ControlSequence, &str)> {
    tuple((parse_control_sequence, parse_non_control_code))(input)
}

#[derive(Clone, Copy)]
struct TerminalColors {
    foreground: Rgb<u8>,
    background: Rgb<u8>,
}

impl Debug for TerminalColors {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_fmt(format_args!(
            "(#{:02X}{:02X}{:02X}, #{:02X}{:02X}{:02X})",
            self.foreground.0[0],
            self.foreground.0[1],
            self.foreground.0[2],
            self.background.0[0],
            self.background.0[1],
            self.background.0[2],
        ))
    }
}

impl Default for TerminalColors {
    fn default() -> Self {
        Self {
            foreground: Rgb([255_u8; 3_usize]),
            background: Rgb([0_u8; 3_usize]),
        }
    }
}

#[derive(Clone, Debug)]
struct TextCol<'s> {
    text: &'s str,
    col: TerminalColors,
}

fn iter_text_cols(vim_str: &str) -> impl Iterator<Item = TextCol> {
    let mut col: TerminalColors = TerminalColors::default();

    let mut pairs: VecDeque<(ControlSequence, &str)> =
        iterator(vim_str, parse_control_sequence_then_text).collect();

    let start: usize = pairs
        .iter()
        .position(|(control_sequence, _)| {
            *control_sequence == ControlSequence::CursorPosition(UVec2::ONE)
        })
        .map_or(pairs.len(), |reset_position_index| reset_position_index + 1);

    for (control_sequence, _) in pairs.drain(..start) {
        control_sequence.apply(&mut col);
    }

    pairs
        .into_iter()
        .filter_map(move |(control_sequence, text)| {
            control_sequence.apply(&mut col);

            let text: &str = text.trim();

            if text.is_empty() || !control_sequence.proceeds_text() {
                None
            } else {
                Some((text, col))
            }
        })
        .flat_map(|(text, col)| {
            text.split_ascii_whitespace()
                .map(move |text| TextCol { text, col })
        })
}

#[derive(Clone, Debug)]
struct TextPosCol<'s> {
    text: &'s str,
    pos: Pos,
    col: TerminalColors,
}

#[derive(Debug, Default, Hash, Clone)]
struct ZipperState {
    line_y: u32,
    line_text_pos_index: usize,
    text_pos_index: usize,
    text_pos_byte_index: usize,
    text_col_index: usize,
    zipper_node_index: Option<usize>,
    final_candidate: bool,
}

impl ZipperState {
    fn try_get_text_pos<'s>(&self, text_poses: &[TextPos<'s>]) -> Option<TextPos<'s>> {
        text_poses.get(self.text_pos_index).map(|text_pos| TextPos {
            text: &text_pos.text[self.text_pos_byte_index..],
            pos: Pos(UVec2::new(
                text_pos.pos.0.x + self.text_pos_byte_index as u32,
                text_pos.pos.0.y,
            )),
        })
    }

    fn try_get_text_col<'s>(&self, text_cols: &[TextCol<'s>]) -> Option<TextCol<'s>> {
        text_cols.get(self.text_col_index).cloned()
    }

    fn hash(&self) -> u64 {
        let mut hasher: DefaultHasher = DefaultHasher::new();

        Hash::hash(&self, &mut hasher);

        hasher.finish()
    }

    fn step(&mut self, text_poses: &[TextPos], len: usize) {
        self.text_pos_byte_index += len;

        if self.try_get_text_pos(text_poses).unwrap().text.is_empty() {
            self.text_pos_index += 1_usize;
            self.text_pos_byte_index = 0_usize;
        }
    }
}

#[derive(Debug)]
struct ZipperNode<'s> {
    text_pos_col: TextPosCol<'s>,
    prev_zipper_node_index: Option<usize>,
}

fn zipper<'s>(text_poses: &[TextPos<'s>], text_cols: &[TextCol<'s>]) -> Vec<TextPosCol<'s>> {
    let ends_in_double = |text: &str| {
        let text_len: usize = text.len();
        let text_bytes: &[u8] = text.as_bytes();

        text_len >= 2_usize && text_bytes[text_len - 2_usize] == text_bytes[text_len - 1_usize]
    };

    let push_zipper_node = |text: &'s str,
                            text_pos: &TextPos,
                            text_col: &TextCol,
                            zipper_state: &ZipperState,
                            final_candidate: bool,
                            zipper_nodes: &mut Vec<ZipperNode<'s>>,
                            zipper_state_queue: &mut VecDeque<ZipperState>,
                            zipper_state_hashes: &mut HashSet<u64>| {
        let zipper_node_index: Option<usize> = Some(zipper_nodes.len());

        zipper_nodes.push(ZipperNode {
            text_pos_col: TextPosCol {
                text,
                pos: text_pos.pos,
                col: text_col.col,
            },
            prev_zipper_node_index: zipper_state.zipper_node_index,
        });

        let mut zipper_state: ZipperState = zipper_state.clone();

        /* Adjust the line information before stepping the `ZipperState` in case the next */
        /* `TextPos` text is just "@", as it'll pertain to the current line, not the next line. */
        if zipper_state.line_y != text_pos.pos.0.y {
            zipper_state.line_y = text_pos.pos.0.y;
            zipper_state.line_text_pos_index = zipper_state.text_pos_index;
        }

        zipper_state.step(text_poses, text.len());
        zipper_state.text_col_index += 1_usize;
        zipper_state.zipper_node_index = zipper_node_index;
        zipper_state.final_candidate = final_candidate;

        if zipper_state_hashes.insert(zipper_state.hash()) {
            zipper_state_queue.push_back(zipper_state);
        }
    };

    let construct_text_pos_cols = |zipper_nodes: &[ZipperNode<'s>]| -> Vec<TextPosCol<'s>> {
        let mut text_pos_cols: VecDeque<TextPosCol> = VecDeque::new();
        let mut zipper_node: Option<&ZipperNode> = zipper_nodes.last();

        while let Some(zipper_node_ref) = zipper_node {
            text_pos_cols.push_front(zipper_node_ref.text_pos_col.clone());
            zipper_node = zipper_node_ref
                .prev_zipper_node_index
                .map(|prev_zipper_node_index| &zipper_nodes[prev_zipper_node_index]);
        }

        text_pos_cols.into()
    };

    let mut zipper_nodes: Vec<ZipperNode> = Vec::new();

    let zipper_state_default: ZipperState = ZipperState::default();

    let mut zipper_state_hashes: HashSet<u64> = [zipper_state_default.hash()].into_iter().collect();
    let mut zipper_state_queue: VecDeque<ZipperState> = vec![zipper_state_default].into();

    while let Some(zipper_state) = zipper_state_queue.pop_front() {
        if zipper_state.final_candidate && zipper_state.text_pos_index == text_poses.len() {
            return construct_text_pos_cols(&zipper_nodes);
        } else if let Some((text_pos, text_col)) = zipper_state
            .try_get_text_pos(text_poses)
            .zip(zipper_state.try_get_text_col(text_cols))
        {
            if text_pos.text.starts_with(text_col.text) {
                push_zipper_node(
                    text_col.text,
                    &text_pos,
                    &text_col,
                    &zipper_state,
                    true,
                    &mut zipper_nodes,
                    &mut zipper_state_queue,
                    &mut zipper_state_hashes,
                );
            }

            /* Sometimes, when Vim is printing out a really long line, it'll print a double of a */
            /* character before breaking to a new line. We need to consider that possibility. */
            if ends_in_double(text_col.text) {
                let text: &str = &text_col.text[..text_col.text.len() - 1_usize];

                if text_pos.text.starts_with(text) {
                    push_zipper_node(
                        text,
                        &text_pos,
                        &text_col,
                        &zipper_state,
                        false,
                        &mut zipper_nodes,
                        &mut zipper_state_queue,
                        &mut zipper_state_hashes,
                    );
                }
            }

            /* Sometimes, when Vim is printing out a long line, it prints out an erroneous '@', */
            /* then jumps back to the start of the line and prints out the full line. I believe */
            /* this is a buffering artifact, where the print cursor gets caught up to where Vim */
            /* has loaded in, and so it jumps back to give it time to read the remaining parts of */
            /* the line, but this is just speculation. In any case, if we found a `TextCol` */
            /* that's just "@", we need to find the zipper node index corresponding to a */
            /* different line than what we're currently on, and reset our `TextPos` identifying */
            /* data to reflect that. */
            if text_col.text == "@" {
                let mut zipper_node_index: Option<usize> = zipper_state.zipper_node_index;

                while let Some(zipper_node) =
                    zipper_node_index.map(|zipper_node_index| &zipper_nodes[zipper_node_index])
                {
                    if zipper_node.text_pos_col.pos.0.y == zipper_state.line_y {
                        zipper_node_index = zipper_node.prev_zipper_node_index;
                    } else {
                        break;
                    }
                }

                let mut zipper_state: ZipperState = zipper_state.clone();

                zipper_state.text_pos_index = zipper_state.line_text_pos_index;
                zipper_state.text_pos_byte_index = 0_usize;
                zipper_state.text_col_index += 1_usize;
                zipper_state.zipper_node_index = zipper_node_index;
                zipper_state.final_candidate = false;

                if zipper_state_hashes.insert(zipper_state.hash()) {
                    zipper_state_queue.push_back(zipper_state);
                }
            }
        }
    }

    /* This is a failure outcome, but we should try to find the node that was the closest to */
    /* being complete for debugging purposes. */
    construct_text_pos_cols(&zipper_nodes)
}

fn debug_print_to_file<D: Debug>(debug: &D, file_path: &str) {
    File::create(file_path)
        .unwrap()
        .write_fmt(format_args!("{debug:#?}"))
        .unwrap();
}

pub struct Base64;

impl Base64 {
    pub const OFFSET_1: u8 = b'a' - 26_u8;
    pub const OFFSET_2: u8 = 52_u8 - b'0';
    pub const BITS_PER_BYTE: usize = 6_usize;

    fn decode(value: u8) -> u8 {
        match value {
            b'A'..=b'Z' => value - b'A',
            b'a'..=b'z' => value - Self::OFFSET_1,
            b'0'..=b'9' => value + Self::OFFSET_2,
            b'+' => 62_u8,
            b'/' => 63_u8,
            _ => 0_u8,
        }
    }
}

#[derive(Clone, PartialEq)]
pub struct CharAlphaGrid(pub [u8; CharAlphaGrid::PIXEL_AREA]);

impl CharAlphaGrid {
    pub const PIXEL_WIDTH: usize = 16_usize;
    pub const PIXEL_HEIGHT: usize = 32_usize;
    pub const PIXEL_AREA: usize = Self::PIXEL_WIDTH * Self::PIXEL_HEIGHT;
    pub const BITS: u32 = 6_u32;
    pub const MAX: u8 = ((1_u32 << Self::BITS) - 1_u32) as u8;
    pub const SCALE: f32 = u8::MAX as f32 / Self::MAX as f32;

    fn alpha(drawing: u8, parity: u8) -> u8 {
        (((drawing & Self::MAX) ^ Base64::decode(parity)) as f32 * Self::SCALE).round() as u8
    }
}

impl Default for CharAlphaGrid {
    fn default() -> Self {
        /* SAFETY: `Self` is just an array of `u8`s, for which zeroed bytes are valid. */
        unsafe { MaybeUninit::zeroed().assume_init() }
    }
}

#[derive(PartialEq)]
pub struct CharAlphaGridArr(pub Box<[CharAlphaGrid; CharAlphaGridArr::PRINTABLE_LEN]>);

#[derive(Debug)]
pub enum PrintError {
    FailedToReadTextFile(IoError),
    FailedToGetVimString(String),
    NonAsciiChar(char),
    FailedToSaveImageFile(ImageError),
}

impl CharAlphaGridArr {
    pub const PRINTABLE_RANGE: Range<char> = '\x20'..'\x7F';
    pub const PRINTABLE_LEN: usize = Self::PRINTABLE_RANGE.end as usize
        - Self::PRINTABLE_RANGE.start as usize
        /* Add one so the information represented is an easier number to work with (96, not 95) */
        + 1_usize;

    pub const BLOCK_BYTE_LEN: usize = 32_usize;
    pub const FRAME_BLOCK_LEN: usize = 14_usize;
    pub const FRAME_BYTE_LEN: usize =
        CharAlphaGridArr::BLOCK_BYTE_LEN * CharAlphaGridArr::FRAME_BLOCK_LEN;

    /* The number of payload bits in the manifest (which is also the number of blocks in */
    /* `BYTES`). This is the number of pixels needed to print, for each character in the */
    /* printable range, times two (the alpha of each pixel is encoded in the lower 6 bits of a */
    /* drawing byte XORed with a base-64 parity byte), divided by the number of blocks per byte. */
    pub const MANIFEST_BIT_LEN: usize =
        CharAlphaGrid::PIXEL_AREA * CharAlphaGridArr::PRINTABLE_LEN * 2_usize
            / CharAlphaGridArr::BLOCK_BYTE_LEN;
    pub const MANIFEST_BYTE_LEN: usize = CharAlphaGridArr::padded_byte_len(
        CharAlphaGridArr::MANIFEST_BIT_LEN / Base64::BITS_PER_BYTE,
    );
    pub const NEW_LINE_SLICE: &'static [u8] = "\n ".as_bytes();
    pub const NEW_LINE_SLICE_LEN: usize = CharAlphaGridArr::NEW_LINE_SLICE.len();

    pub const fn padded_byte_len(byte_len: usize) -> usize {
        byte_len
            + (byte_len / CharAlphaGridArr::FRAME_BYTE_LEN + 1_usize)
                * CharAlphaGridArr::NEW_LINE_SLICE_LEN
    }

    pub fn from_image_bytes(bytes: &[u8]) -> Self {
        let mut char_alpha_grids: Self = Self::default();
        let mut manifest: BitVec<u8, Lsb0> = BitVec::with_capacity(Self::MANIFEST_BIT_LEN);

        /* Construct the manifest from the base-64-encoded bytes at the end of `BYTES`. */
        for manifest_byte in bytes[bytes.len() - Self::MANIFEST_BYTE_LEN..].iter() {
            if !Self::NEW_LINE_SLICE.contains(manifest_byte) {
                manifest.extend_from_bitslice(
                    &[Base64::decode(*manifest_byte)].as_bits::<Lsb0>()[..Base64::BITS_PER_BYTE],
                );
            }
        }

        /* For a given block index, iter over its bytes. */
        let iter_bytes = |block_index| {
            let byte_index: usize = Self::padded_byte_len(block_index * Self::BLOCK_BYTE_LEN);

            bytes[byte_index..byte_index + Self::BLOCK_BYTE_LEN]
                .iter()
                .copied()
        };

        /* Any 1s in the manifest correspond to drawing blocks. */
        let iter_drawing_bytes = manifest.iter_ones().flat_map(iter_bytes);

        /* Any 0s in the manifest correspond to parity blocks. */
        let iter_parity_bytes = manifest.iter_zeros().flat_map(iter_bytes);
        let iter_alpha_bytes = char_alpha_grids
            .0
            .iter_mut()
            .flat_map(|char_alpha_grid| char_alpha_grid.0.iter_mut());

        for ((drawing, parity), alpha) in iter_drawing_bytes
            .zip(iter_parity_bytes)
            .zip(iter_alpha_bytes)
        {
            *alpha = CharAlphaGrid::alpha(drawing, parity);
        }

        char_alpha_grids
    }
}

impl Default for CharAlphaGridArr {
    fn default() -> Self {
        Self(alloc_zeroed_box())
    }
}

fn print(char_alpha_grid_arr: &CharAlphaGridArr, args: &Args) -> Result<(), PrintError> {
    use PrintError as Error;

    const CHAR_PIXEL_WIDTH: usize = CharAlphaGrid::PIXEL_WIDTH;
    const CHAR_PIXEL_HEIGHT: usize = CharAlphaGrid::PIXEL_HEIGHT;

    let file_string: String =
        read_to_string(&args.input_text_path).map_err(Error::FailedToReadTextFile)?;
    let (image_pixel_width, image_pixel_height): (usize, usize) = file_string
        .lines()
        .try_fold((0_usize, 0_usize), |(char_width, char_height), line| {
            if let Some(non_ascii_char) = line.chars().find(|c| !c.is_ascii()) {
                Err(non_ascii_char)
            } else {
                Ok((char_width.max(line.len()), char_height + 1))
            }
        })
        .map(|(char_width, char_height)| {
            (
                (char_width * CHAR_PIXEL_WIDTH).max(args.min_pixel_width),
                (char_height * CHAR_PIXEL_HEIGHT).max(args.min_pixel_height),
            )
        })
        .map_err(Error::NonAsciiChar)?;
    let vim_string: String =
        try_get_vim_string(&args.input_text_path).map_err(Error::FailedToGetVimString)?;

    if args.debug {
        debug_print_to_file(
            &(&mut iterator::<&str, (char, &str), NomError<&str>, _>(
                vim_string.as_str(),
                tuple((
                    satisfy(is_control_code),
                    take_while(|c: char| !is_control_code(c)),
                )),
            ))
                .collect::<Vec<(char, &str)>>(),
            output_file!("vim_string.txt"),
        );

        let tps: Vec<TextPos> = iter_text_poses(&file_string, args.tab_size).collect();

        debug_print_to_file(&tps, output_file!("text_poses.txt"));

        let tcs: Vec<TextCol> = iter_text_cols(&vim_string).collect();

        debug_print_to_file(&tcs, output_file!("text_cols.txt"));

        let tpcs: Vec<TextPosCol> = zipper(&tps, &tcs);

        debug_print_to_file(&tpcs, output_file!("text_pos_cols.txt"));

        if args.skip_print {
            return Ok(());
        }
    }

    let mut image: RgbaImage = RgbaImage::new(image_pixel_width as u32, image_pixel_height as u32);

    const CHANNEL_COUNT: usize = Rgba::<u8>::CHANNEL_COUNT as usize;

    let background: Rgba<u8> =
        (&mut iterator(vim_string.as_str(), parse_control_sequence_then_text))
            .find(|(control_sequence, _)| {
                matches!(control_sequence, ControlSequence::SgrSetBackground(_))
            })
            .map_or(
                Rgba([0_u8, 0_u8, 0_u8, u8::MAX]),
                |(control_sequence, _)| match control_sequence {
                    ControlSequence::SgrSetBackground(background_rgb) => {
                        let mut background_rgba: Rgba<u8> = background_rgb.to_rgba();

                        *background_rgba.0.last_mut().unwrap() = u8::MAX;

                        background_rgba
                    }
                    _ => panic!("unreachable"),
                },
            );

    for pixel in image.pixels_mut() {
        *pixel = background;
    }

    let text_poses: Vec<TextPos> = iter_text_poses(&file_string, args.tab_size).collect();
    let text_cols: Vec<TextCol> = iter_text_cols(&vim_string).collect();
    let total_printed_bytes: usize = text_poses.iter().map(|text_pos| text_pos.text.len()).sum();
    let printed_bytes_factor: f32 = 100.0_f32 / total_printed_bytes as f32;
    let mut printed_bytes: usize = 0_usize;
    let mut prev_percentage: u8 = 0_u8;
    let mut char_alpha_grid_indices: Vec<usize> = Vec::new();

    for TextPosCol { text, pos, col } in zipper(&text_poses, &text_cols) {
        /* Collect the indices of the characters to be printed. */
        char_alpha_grid_indices.extend(text.as_bytes().iter().copied().map(|byte| {
            byte.saturating_sub(CharAlphaGridArr::PRINTABLE_RANGE.start as u8) as usize
        }));

        let image_x_start: usize = pos.0.x as usize * CHAR_PIXEL_WIDTH;
        let image_x_end: usize = image_x_start + text.len() * CHAR_PIXEL_WIDTH;
        let image_y_start: usize = pos.0.y as usize * CHAR_PIXEL_HEIGHT;

        /* Loop over the pixel rows to be printed. */
        for char_y in 0_usize..CHAR_PIXEL_HEIGHT {
            let image_y: usize = char_y + image_y_start;
            let alpha_start: usize = char_y * CHAR_PIXEL_WIDTH;
            let alpha_range: Range<usize> = alpha_start..alpha_start + CHAR_PIXEL_WIDTH;
            let byte_index = |image_x: usize| -> usize {
                (image_y * image_pixel_width + image_x) * CHANNEL_COUNT
            };

            /* Loop over the pixels and alpha values within the row. */
            for (pixel, alpha) in (*image)[byte_index(image_x_start)..byte_index(image_x_end)]
                .chunks_exact_mut(CHANNEL_COUNT)
                .map(<Rgba<u8> as Pixel>::from_slice_mut)
                .zip(
                    char_alpha_grid_indices
                        .iter()
                        .flat_map(|char_alpha_grid_index| {
                            char_alpha_grid_arr.0[*char_alpha_grid_index].0[alpha_range.clone()]
                                .iter()
                                .copied()
                        }),
                )
            {
                pixel.0[..col.background.0.len()].copy_from_slice(&col.background.0);

                let mut rgba: Rgba<u8> = col.foreground.to_rgba();

                *rgba.0.last_mut().unwrap() = alpha;

                pixel.blend(&rgba);
            }
        }

        printed_bytes += text.len();

        let curr_percentage: u8 = (printed_bytes as f32 * printed_bytes_factor) as u8;

        if curr_percentage != prev_percentage {
            println!("{curr_percentage}% done printing image");

            prev_percentage = curr_percentage;
        }

        char_alpha_grid_indices.clear();
    }

    image
        .save(&args.output_image_path)
        .map_err(Error::FailedToSaveImageFile)
}

#[derive(Debug, Parser)]
#[clap(disable_help_flag = true)]
struct Args {
    #[arg(short, long, default_value_t = IMAGE_TEXT_PATH.into())]
    input_text_path: String,

    #[arg(short, long, default_value_t = OUTPUT_IMAGE_PATH.into())]
    output_image_path: String,

    #[arg(short = 'w', long, default_value_t)]
    min_pixel_width: usize,

    #[arg(short = 'h', long, default_value_t)]
    min_pixel_height: usize,

    #[arg(short, long, default_value_t = TAB_SIZE)]
    tab_size: usize,

    #[arg(short, long)]
    debug: bool,

    #[arg(short, long)]
    skip_print: bool,

    #[clap(long, action = HelpLong)]
    help: (),
}

#[allow(dead_code)]
fn main() {
    let args: Args = Args::parse();

    if args.output_image_path == OUTPUT_IMAGE_PATH {
        create_dir_all(output_file!("")).ok();
    }

    let char_alpha_grid_arr: CharAlphaGridArr = CharAlphaGridArr::from_image_bytes(BYTES);

    if let Err(error) = print(&char_alpha_grid_arr, &args) {
        eprintln!("Encountered error attempting to print with args {args:#?}\n{error:#?}");
    }
}

/* End copy section */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_zipper() {
        const FILE_PATH: &str = "./examples/drawing_hands.rs";

        let file_string: String = read_to_string(FILE_PATH).unwrap();
        let vim_string: String = try_get_vim_string(FILE_PATH).unwrap();
        let text_poses: Vec<TextPos> = iter_text_poses(&file_string, TAB_SIZE).collect();
        let text_cols: Vec<TextCol> = iter_text_cols(&vim_string).collect();
        let text_pos_cols: Vec<TextPosCol> = zipper(&text_poses, &text_cols);

        dbg!(text_pos_cols);
    }
}
