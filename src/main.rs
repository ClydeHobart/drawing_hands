use {
    ab_glyph::{Font, FontRef, InvalidFont, OutlineCurve},
    ab_glyph_rasterizer::{Point, Rasterizer},
    bitvec::prelude::*,
    clap::Parser,
    content::*,
    image::{imageops::FilterType, open, DynamicImage, GrayImage, ImageError, ImageResult},
    parse::into_tokens,
    std::{
        fmt::{Debug, Formatter, Result as FmtResult},
        fs::read,
        io::Error as IoError,
        iter::Peekable,
        mem::MaybeUninit,
        ops::{Add, Mul, RangeInclusive, RangeTo},
        path::Path,
        process::{Command, Stdio},
        slice::ChunksExact,
        str::from_utf8_unchecked,
    },
    ttf_parser::{Face, FaceParsingError},
    write::try_write_example,
};

#[macro_use]
mod content;

/// Code for parsing writable tokens that can freely have whitespace inserted between them. This
/// its used by `TextBuffer::write_tokens` to tightly pack tokens, filling as much of the line as
/// possible. Note that due to the many robustness errors in this code, it is not intended to be
/// used as is outside of this project to parse what it claims to be able to parse at face value:
/// the many known constraints and limitations of it work for this use case because it only needs
/// to work for the static text that is the `content` module.
mod parse {
    use nom::{
        branch::alt,
        bytes::complete::{tag, take, take_until, take_while, take_while1},
        character::complete::{line_ending, not_line_ending, space0},
        combinator::{iterator, map, recognize},
        sequence::{preceded, terminated, tuple},
        IResult,
    };

    /// Yields a single line.
    fn parse_line(input: &str) -> IResult<&str, &str> {
        map(
            tuple((space0, not_line_ending, line_ending)),
            |(_, line, _)| line,
        )(input)
    }

    /// Yields a `str` that doesn't contain `'"'`.
    fn parse_quoteless_str(input: &str) -> IResult<&str, &str> {
        take_while(|c: char| c != '"')(input)
    }

    /// Yields a string literal.
    fn parse_string_literal(input: &str) -> IResult<&str, &str> {
        recognize(tuple((tag("\""), parse_quoteless_str, tag("\""))))(input)
    }

    /// Yields a char literal. Note that this doesn't properly handle `'\''`. This also will eat
    /// invalid char literals that contain multiple characters between the `'`s.
    fn parse_char_literal(input: &str) -> IResult<&str, &str> {
        recognize(tuple((
            tag("'"),
            take_while1(|c: char| c != '\''),
            tag("'"),
        )))(input)
    }

    /// Yields a byte literal. Note that this doesn't properly handle `b'\''`. This also will eat
    /// invalid byte literals that contain multiple characters between the `'`s.
    fn parse_byte_literal(input: &str) -> IResult<&str, &str> {
        recognize(tuple((tag("b"), parse_char_literal)))(input)
    }

    /// Yields a `/* block comment */`
    fn parse_block_comment(input: &str) -> IResult<&str, &str> {
        recognize(tuple((tag("/*"), take_until("*/"), tag("*/"))))(input)
    }

    /// Yields an identifier, a keyword, or a numeric literal. This will also yield invalid tokens
    /// that start with a numeric character followed by characters that don't form a numeric literal
    /// together (like `0foo`).
    fn parse_identifier_keyword_or_numeric_literal(input: &str) -> IResult<&str, &str> {
        take_while1(|c: char| c.is_ascii_alphanumeric() || c == '_')(input)
    }

    /// Yields a label or lifetime. Note that this must be processed after attempting
    /// `parse_char_literal`. Due to the issues with `parse_identifier_keyword_or_numeric_literal`,
    /// this will also match invalid labels.
    fn parse_label_or_lifetime(input: &str) -> IResult<&str, &str> {
        recognize(preceded(
            tag("'"),
            parse_identifier_keyword_or_numeric_literal,
        ))(input)
    }

    /// Yields the smallest token that can be written without worrying about whitespace being
    /// inserted.
    fn parse_token(input: &str) -> IResult<&str, &str> {
        macro_rules! alt_tags { ( $( $tag:literal ),+ ) => { alt(( $( tag($tag), )+ )) }; }

        terminated(
            alt((
                // Parse multi-char operators. Taken from
                // https://doc.rust-lang.org/book/appendix-02-operators.html.
                alt((
                    // 1. 3-char literals
                    alt_tags!("..=", "<<=", ">>="),
                    // 2. Bitshift
                    alt_tags!("<<", ">>"),
                    // 3. Comparison & boolean operation
                    alt_tags!("==", "!=", "<=", ">=", "&&", "||"),
                    // 4. Assignment
                    alt_tags!("+=", "-=", "*=", "/=", "%=", "&=", "|=", "^="),
                    // 5. Other
                    alt_tags!("::", "->", "=>", ".."),
                )),
                parse_string_literal,
                parse_char_literal,
                parse_byte_literal,
                parse_block_comment,
                parse_label_or_lifetime,
                parse_identifier_keyword_or_numeric_literal,
                take(1_usize),
            )),
            space0,
        )(input)
    }

    pub fn into_tokens(input: &str) -> Vec<&str> {
        let mut tokens: Vec<&str> = Vec::new();

        for line in &mut iterator(input, parse_line).filter(|line| !line.is_empty()) {
            for token in &mut iterator(line, parse_token) {
                tokens.push(token);
            }
        }

        tokens
    }

    #[test]
    fn test_into_tokens() {
        const TEST_INTO_TOKENS: &str = r#"fn main() {
    println!("this is a literal inside a macro inside a func");

    let x: i32 = 1;

    dbg!(x);
}

#[cfg(test)]
mod test {
    #[test]
    fn some_test_0() {
        let this_is_a_really_long_variable_name: i32 = 999999;

        let my_str = "this is a another str as well as the one in main()";

        println!("{my_str} {this_is_a_really_long_variable_name}");

        println!("{} {} {}", "a", "b", "c");
    }
}
"#;

        assert_eq!(
            into_tokens(TEST_INTO_TOKENS),
            vec![
                "fn",
                "main",
                "(",
                ")",
                "{",
                "println",
                "!",
                "(",
                "\"this is a literal inside a macro inside a func\"",
                ")",
                ";",
                "let",
                "x",
                ":",
                "i32",
                "=",
                "1",
                ";",
                "dbg",
                "!",
                "(",
                "x",
                ")",
                ";",
                "}",
                "#",
                "[",
                "cfg",
                "(",
                "test",
                ")",
                "]",
                "mod",
                "test",
                "{",
                "#",
                "[",
                "test",
                "]",
                "fn",
                "some_test_0",
                "(",
                ")",
                "{",
                "let",
                "this_is_a_really_long_variable_name",
                ":",
                "i32",
                "=",
                "999999",
                ";",
                "let",
                "my_str",
                "=",
                "\"this is a another str as well as the one in main()\"",
                ";",
                "println",
                "!",
                "(",
                "\"{my_str} {this_is_a_really_long_variable_name}\"",
                ")",
                ";",
                "println",
                "!",
                "(",
                "\"{} {} {}\"",
                ",",
                "\"a\"",
                ",",
                "\"b\"",
                ",",
                "\"c\"",
                ")",
                ";",
                "}",
                "}"
            ]
        );
    }
}

mod write {
    use {
        super::*,
        chrono::prelude::*,
        static_assertions::const_assert,
        std::{
            fs::{create_dir_all, read_dir, remove_file, write, DirEntry},
            io::Write,
        },
    };

    const CONTENT: &str = include_str!("content.rs");

    const_assert!(CONTENT.is_ascii());

    const EXAMPLES_PATH: &str = "./examples/";

    const TAB_SIZE_IC: InsertionContext =
        InsertionContext::new("const TAB_SIZE: usize = 4_usize;", 8_usize, 7_usize);
    const IMAGE_TEXT_PATH_IC: InsertionContext =
        InsertionContext::new(r#"const IMAGE_TEXT_PATH: &str = "";"#, 2_usize, 2_usize);
    const OUTPUT_IMAGE_PATH_IC: InsertionContext = InsertionContext::new(
        r#"const OUTPUT_IMAGE_PATH: &str = output_file!("");"#,
        3_usize,
        3_usize,
    );
    const BYTES_IC: InsertionContext =
        InsertionContext::new(r#"const BYTES: &[u8] = br"";"#, 3_usize, 1_usize);

    const END_COPY_SECTION_PATTERN: &str = "/* End copy section */";
    #[derive(Default)]
    struct TextBuffer {
        bytes: Vec<u8>,
        tab_size: usize,
        column: usize,
        end_column: usize,
    }

    impl TextBuffer {
        const END_COMMENT: &'static str = " */";
        const START_COMMENT: &'static str = "/*";

        fn new(capacity: usize, tab_size: usize, end_column: usize) -> Self {
            Self {
                bytes: Vec::with_capacity(capacity),
                tab_size,
                column: 0_usize,
                end_column,
            }
        }

        fn write(&mut self, text: &str) {
            self.bytes.reserve(text.len());

            for byte in text.as_bytes().iter().copied() {
                self.bytes.push(byte);

                match byte {
                    b'\t' => {
                        self.column += self.tab_size - self.column % self.tab_size;
                    }
                    b'\n' => {
                        self.column = 0_usize;
                    }
                    _ => {
                        self.column += 1_usize;
                    }
                }
            }
        }

        fn is_punctuation(byte: u8) -> bool {
            let c: char = byte as char;

            !c.is_ascii_alphanumeric() && c != '_'
        }

        fn write_token(&mut self, token: &str) {
            if let Some((last_old_byte, first_new_byte)) = self
                .bytes
                .last()
                .copied()
                .zip(token.as_bytes().first().copied())
            {
                let is_continuing_comment: bool =
                    self.bytes.ends_with(Self::END_COMMENT.as_bytes())
                        && token.starts_with(Self::START_COMMENT);

                let token: &str = if is_continuing_comment {
                    self.bytes
                        .truncate(self.bytes.len() - Self::END_COMMENT.len());
                    self.column -= Self::END_COMMENT.len();

                    &token[Self::START_COMMENT.len()..]
                } else {
                    token
                };

                let is_space_needed: bool = !is_continuing_comment
                    && !Self::is_punctuation(last_old_byte)
                    && !Self::is_punctuation(first_new_byte);

                if self.column + is_space_needed as usize + token.len() <= self.end_column {
                    if is_space_needed {
                        self.write(" ");
                    }

                    self.write(token);
                } else {
                    self.write("\n");
                    self.write(token);
                }
            } else {
                self.write(token);
            }
        }

        fn write_tokens(&mut self, text: &str) {
            let mut comment_buffer: Vec<u8> = Vec::new();

            for token in into_tokens(text) {
                if token.starts_with(Self::START_COMMENT) && token.ends_with(Self::END_COMMENT) {
                    for token in token
                        [Self::START_COMMENT.len()..token.len() - Self::END_COMMENT.len()]
                        .split_ascii_whitespace()
                    {
                        comment_buffer.clear();
                        write!(&mut comment_buffer, "/* {token} */").ok();

                        // SAFETY: `comment_buffer` only ever contains bytes written by the
                        // `write!` macro.
                        self.write_token(unsafe { from_utf8_unchecked(&comment_buffer) });
                    }
                } else {
                    self.write_token(token);
                }
            }
        }
    }

    struct InsertionContext {
        /// The pattern to search for in `CONTENT`.
        pattern: &'static str,

        /// How many bytes from the end of the pattern should the insertion start.
        start_insertion_trailing: usize,

        /// How many bytes from the end of the pattern should the insertion end.
        end_insertion_trailing: usize,
    }

    impl InsertionContext {
        const fn new(
            pattern: &'static str,
            start_insertion_trailing: usize,
            end_insertion_trailing: usize,
        ) -> Self {
            Self {
                pattern,
                start_insertion_trailing,
                end_insertion_trailing,
            }
        }

        fn write<'s, I: IntoIterator<Item = &'s str>>(
            &self,
            insertions: I,
            text_buffer: &mut TextBuffer,
        ) {
            let start_pattern: usize = self.start_pattern();
            let end_pattern: usize = start_pattern + self.pattern.len();
            let start_insert: usize = end_pattern - self.start_insertion_trailing;
            let end_insert: usize = end_pattern - self.end_insertion_trailing;

            text_buffer.write(&CONTENT[start_pattern..start_insert]);

            for insertion in insertions {
                text_buffer.write(insertion);
            }

            text_buffer.write(&CONTENT[end_insert..end_pattern]);
            text_buffer.write("\n");
        }

        fn start_pattern(&self) -> usize {
            find_start_pattern(self.pattern)
        }

        fn end_pattern(&self) -> usize {
            find_end_pattern(self.pattern)
        }
    }

    fn remove_vim_txt_cache_files() {
        for vim_txt_dir_entry in read_dir(output_file!())
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|dir_entry| dir_entry.ok())
            .filter(|dir_entry| {
                dir_entry
                    .file_name()
                    .to_str()
                    .map_or(false, |file_name| file_name.ends_with(".vim.txt"))
            })
            .collect::<Vec<DirEntry>>()
        {
            remove_file(vim_txt_dir_entry.path()).ok();
        }
    }

    fn find_start_pattern(pattern: &str) -> usize {
        CONTENT.find(pattern).unwrap()
    }

    fn find_end_pattern(pattern: &str) -> usize {
        find_start_pattern(pattern) + pattern.len()
    }

    fn get_signature() -> String {
        let basic_signature: String = format!(
            "github.com/ClydeHobart/drawing_hands, {}",
            Local::now().date_naive().format("%Y-%m-%d")
        );

        Command::new("figlet")
            .args(["-f", "slant", "-w", "500", &basic_signature])
            .output()
            .ok()
            .and_then(|output| {
                output
                    .status
                    .success()
                    .then(|| String::from_utf8(output.stdout).ok())
            })
            .flatten()
            .unwrap_or(basic_signature)
    }

    #[derive(Debug)]
    pub enum WriteExampleError {
        ImageBytesDataError(ImageBytesDataError),
        InvalidExampleChar(char),
        FailedToCreateDir(IoError),
        FailedToWriteExample(IoError),
    }

    pub fn try_write_example(
        font_path: &str,
        image_path: &str,
        example_name: &str,
        tab_size: usize,
    ) -> Result<(), WriteExampleError> {
        use WriteExampleError as Error;

        remove_vim_txt_cache_files();

        let image_bytes_data: ImageBytesData =
            ImageBytesData::try_from_file_paths(font_path, image_path)
                .map_err(Error::ImageBytesDataError)?;

        example_name
            .chars()
            .find(|c| !(c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '_'))
            .map_or(Ok(()), |c| Err(Error::InvalidExampleChar(c)))?;

        create_dir_all(EXAMPLES_PATH).map_err(Error::FailedToCreateDir)?;

        let example_path: String = format!("{EXAMPLES_PATH}{example_name}.rs");
        let octothorpes: String = image_bytes_data.octothorpes_string();
        let image_str: &str = image_bytes_data.bytes_str();

        let mut text_buffer: TextBuffer = TextBuffer::new(
            // This is an overestimate due to whitespace that'll get stripped
            CONTENT.len() + image_str.len(),
            4_usize,
            ImageBytesData::LINE_BYTE_WIDTH,
        );

        text_buffer.write("// Generated by");

        for line in get_signature().lines() {
            text_buffer.write("\n// ");
            text_buffer.write(line);
        }

        text_buffer.write("\n\n");
        text_buffer.write(&CONTENT[..TAB_SIZE_IC.start_pattern()]);
        TAB_SIZE_IC.write([format!("{tab_size}").as_str()], &mut text_buffer);
        IMAGE_TEXT_PATH_IC.write([example_path.as_str()], &mut text_buffer);
        OUTPUT_IMAGE_PATH_IC.write([format!("{}.jpg", example_name).as_str()], &mut text_buffer);
        text_buffer
            .write_tokens(&CONTENT[OUTPUT_IMAGE_PATH_IC.end_pattern()..BYTES_IC.start_pattern()]);
        text_buffer.write("\n");
        BYTES_IC.write(
            [&octothorpes, "\"", image_str, "\"", &octothorpes],
            &mut text_buffer,
        );
        text_buffer.write_tokens(
            &CONTENT[BYTES_IC.end_pattern()..find_start_pattern(END_COPY_SECTION_PATTERN)],
        );
        text_buffer.write("\n");

        write(example_path, text_buffer.bytes).map_err(Error::FailedToWriteExample)
    }
}

const FONT_PATH: &str = "./assets/ubuntu-font-family-0.83/UbuntuMono-R.ttf";
const IMAGE_PATH: &str = "./assets/Drawing_Hands_3x2.png";
const EXAMPLE_NAME: &str = "drawing_hands";
const TAB_SIZE: usize = 4_usize;

#[derive(Clone, Copy)]
struct Vector {
    x: f32,
    y: f32,
}

impl Add<Vector> for Point {
    type Output = Self;

    fn add(self, rhs: Vector) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Mul<Vector> for Point {
    type Output = Self;

    fn mul(self, rhs: Vector) -> Self {
        Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
        }
    }
}

struct CharAlphaGridParams<'a, F: Font> {
    font: &'a F,
    rasterizer: &'a mut Rasterizer,
    offset: Vector,
    scale: Vector,
    c: char,
}

impl<'a, F: Font> CharAlphaGridParams<'a, F> {
    fn draw_curve(&mut self, curve: OutlineCurve) {
        match curve {
            OutlineCurve::Line(p0, p1) => self
                .rasterizer
                .draw_line(self.transform_point(p0), self.transform_point(p1)),
            OutlineCurve::Quad(p0, p1, p2) => self.rasterizer.draw_quad(
                self.transform_point(p0),
                self.transform_point(p1),
                self.transform_point(p2),
            ),
            OutlineCurve::Cubic(p0, p1, p2, p3) => self.rasterizer.draw_cubic(
                self.transform_point(p0),
                self.transform_point(p1),
                self.transform_point(p2),
                self.transform_point(p3),
            ),
        }
    }

    fn transform_point(&self, p: Point) -> Point {
        (p + self.offset) * self.scale
    }

    fn init_char_alpha_grid(&mut self, char_alpha_grid: &mut CharAlphaGrid) -> f32 {
        self.rasterizer.clear();

        if let Some(outline) = self.font.outline(self.font.glyph_id(self.c)) {
            for curve in outline.curves {
                self.draw_curve(curve);
            }

            let mut alpha_sum: f32 = 0.0_f32;

            self.rasterizer.for_each_pixel_2d(|x, y, alpha| {
                alpha_sum += alpha;
                char_alpha_grid.0
                    [CharAlphaGrid::PIXEL_WIDTH * Self::invert_y(y as usize) + x as usize] =
                    (alpha * u8::MAX as f32) as u8;
            });

            alpha_sum / CharAlphaGrid::PIXEL_AREA as f32
        } else {
            char_alpha_grid.0.fill(0_u8);

            0.0_f32
        }
    }

    #[inline(always)]
    fn invert_y(y: usize) -> usize {
        CharAlphaGrid::PIXEL_HEIGHT - y - 1_usize
    }
}

type ByteArray = [u8; 1_usize << u8::BITS];

struct AlphaToByte(Box<ByteArray>);

impl Default for AlphaToByte {
    fn default() -> Self {
        let mut alpha_to_byte: Self = Self(alloc_zeroed_box());

        alpha_to_byte.0.fill(b' ');

        alpha_to_byte
    }
}

struct FontAlphaData {
    char_alpha_grid_arr: CharAlphaGridArr,
    alpha_to_byte: AlphaToByte,
}

#[allow(dead_code)]
#[derive(Debug)]
enum FontAlphaDataError {
    IoError(IoError),
    FaceParsingError(FaceParsingError),
    InvalidFont(InvalidFont),
    MissingGlyph(char),
    MissingHorAdvance(char),
    UnexpectedHorAdvance {
        code_point: char,
        expected: u16,
        actual: u16,
    },
}

impl FontAlphaData {
    fn try_from_file_path<P: AsRef<Path>>(path: P) -> Result<Self, FontAlphaDataError> {
        use FontAlphaDataError as FadError;

        let data: Vec<u8> = read(path).map_err(FadError::IoError)?;
        let face: Face = Face::parse(&data, 0_u32).map_err(FadError::FaceParsingError)?;

        let mut hor_advance: Option<u16> = None;

        // `Font::h_advance_unscaled` uses `Face::glyph_hor_advance` under the hood, but it also
        // panics internally if the information isn't available, without a means to verify the
        // information is available.
        for code_point in CharAlphaGridArr::PRINTABLE_RANGE {
            let Some(glyph_id) = face.glyph_index(code_point) else {
                return Err(FadError::MissingGlyph(code_point));
            };

            let Some(actual) = face.glyph_hor_advance(glyph_id) else {
                return Err(FadError::MissingHorAdvance(code_point));
            };

            if let Some(expected) = hor_advance {
                if actual != expected {
                    return Err(FadError::UnexpectedHorAdvance {
                        code_point,
                        expected,
                        actual,
                    });
                }
            } else {
                hor_advance = Some(actual)
            }
        }

        let font: FontRef = FontRef::try_from_slice(&data).map_err(FadError::InvalidFont)?;
        let mut rasterizer: Rasterizer =
            Rasterizer::new(CharAlphaGrid::PIXEL_WIDTH, CharAlphaGrid::PIXEL_HEIGHT);
        let offset: Vector = Vector {
            x: 0.0_f32,
            y: -face.descender() as f32,
        };
        let scale: Vector = Vector {
            x: CharAlphaGrid::PIXEL_WIDTH as f32 / hor_advance.unwrap() as f32,
            y: CharAlphaGrid::PIXEL_HEIGHT as f32 / face.units_per_em() as f32,
        };
        let mut params: CharAlphaGridParams<FontRef> = CharAlphaGridParams {
            font: &font,
            rasterizer: &mut rasterizer,
            offset,
            scale,
            c: Default::default(),
        };

        let mut char_alpha_grid_arr: CharAlphaGridArr = CharAlphaGridArr::default();

        let mut alpha_char_pairs: Vec<(f32, char)> =
            Vec::with_capacity(CharAlphaGridArr::PRINTABLE_LEN - 1_usize);

        for (index, c) in CharAlphaGridArr::PRINTABLE_RANGE.enumerate() {
            params.c = c;
            alpha_char_pairs.push((
                params.init_char_alpha_grid(&mut char_alpha_grid_arr.0[index]),
                c,
            ));
        }

        alpha_char_pairs.sort_unstable_by(|(alpha_a, _), (alpha_b, _)| alpha_a.total_cmp(alpha_b));

        let ratio: f32 = u8::MAX as f32 / alpha_char_pairs.last().unwrap().0;

        for (alpha, _) in alpha_char_pairs.iter_mut() {
            *alpha *= ratio;
        }

        let mut alpha_to_byte: AlphaToByte = AlphaToByte::default();

        let mut alpha_char_pair_iter: Peekable<_> = alpha_char_pairs.into_iter().peekable();

        'for_loop: for (alpha_index_a, byte) in alpha_to_byte.0.iter_mut().enumerate() {
            let alpha_a: f32 = alpha_index_a as f32;

            while alpha_a
                > if let Some((alpha_b, _)) = alpha_char_pair_iter.peek() {
                    *alpha_b
                } else {
                    break 'for_loop;
                }
            {
                alpha_char_pair_iter.next();
            }

            *byte = alpha_char_pair_iter.peek().unwrap().1 as u8;
        }

        Ok(Self {
            char_alpha_grid_arr,
            alpha_to_byte,
        })
    }
}

#[cfg_attr(test, derive(PartialEq))]
struct ImageBytesData {
    bytes: Vec<u8>,
    octothorpes: usize,
}

#[allow(dead_code)]
#[derive(Debug)]
struct TerminatingSequenceError {
    octothorpe: RangeInclusive<u8>,
    double_quote: RangeInclusive<u8>,
}

#[allow(dead_code)]
#[derive(Debug)]
enum ImageBytesDataError {
    FontAlphaData(FontAlphaDataError),
    Image(ImageError),
    TerminatingSequence(TerminatingSequenceError),
}

impl ImageBytesData {
    const BLOCK_BYTE_WIDTH: usize = CharAlphaGridArr::BLOCK_BYTE_LEN;
    const BLOCK_BYTE_HEIGHT: usize = 16_usize;
    const LONG_BLOCK_LEN: usize = 12_usize;
    const SHORT_BLOCK_LEN: usize = 8_usize;
    const FRAME_BLOCK_WIDTH: usize = CharAlphaGridArr::FRAME_BLOCK_LEN;
    const FRAME_BYTE_WIDTH: usize = CharAlphaGridArr::FRAME_BYTE_LEN;
    const LINE_BYTE_WIDTH: usize = Self::FRAME_BYTE_WIDTH + 2_usize;

    fn try_get_gray_image<P: AsRef<Path>>(path: P) -> ImageResult<GrayImage> {
        let original: DynamicImage = open(path)?;
        let (block_width, block_height): (usize, usize) =
            if original.width() as f32 / original.height() as f32 >= 1.0_f32 {
                (Self::LONG_BLOCK_LEN, Self::SHORT_BLOCK_LEN)
            } else {
                (Self::SHORT_BLOCK_LEN, Self::LONG_BLOCK_LEN)
            };

        Ok(original
            .resize_exact(
                (block_width * Self::BLOCK_BYTE_WIDTH) as u32,
                (block_height * Self::BLOCK_BYTE_HEIGHT) as u32,
                FilterType::CatmullRom,
            )
            .into_luma8())
    }

    fn try_get_octothorpes(
        drawing: &[u8],
        drawing_byte_width: usize,
        alpha_to_byte: &AlphaToByte,
    ) -> Result<usize, TerminatingSequenceError> {
        // SAFETY: `ByteArray` is just an array of bytes, of which `0_u8` is a valid value
        let mut terminating_sequence: ByteArray = unsafe { MaybeUninit::zeroed().assume_init() };

        terminating_sequence[0_usize] = b'"';
        terminating_sequence[1_usize..].fill(b'#');

        let shortest_terminating_sequence: &[u8] = &terminating_sequence[..2_usize];
        let drawing_byte_width_minus_two: usize = drawing_byte_width - 2_usize;

        let candidates: Vec<&[u8]> = drawing
            .chunks_exact(drawing_byte_width)
            .flat_map(|line| (0_usize..drawing_byte_width_minus_two).map(|start| &line[start..]))
            .filter(|candidate| candidate[..2_usize] == *shortest_terminating_sequence)
            .collect();

        if candidates.is_empty() {
            Ok(1_usize)
        } else {
            let mut valid_candidates: BitVec<u32, Lsb0> = bitvec!(u32, Lsb0; 1; candidates.len());
            let mut invalid_candidates: Vec<usize> = Vec::new();

            (3_usize..terminating_sequence.len())
                .find(|octothorpes| {
                    let index: RangeTo<usize> = ..*octothorpes;
                    let terminating_sequence: &[u8] = &terminating_sequence[index];

                    if !valid_candidates.iter_ones().any(|candidate_index| {
                        // Check if the candidate at the specified index is still valid
                        if candidates[candidate_index]
                            .get(index)
                            .filter(|candidate| *candidate == terminating_sequence)
                            .is_some()
                        {
                            // It still matches the terminating sequence
                            true
                        } else {
                            // It no longer matches the terminating sequence
                            invalid_candidates.push(candidate_index);

                            false
                        }
                    }) {
                        // There are no valid candidates for a terminating sequence of b'"' followed
                        // by `*octothorpes`, so we have found what we're looking for
                        true
                    } else {
                        // A valid candidate was found at this length. Mark any candidates that are
                        // no longer valid, either due to being too short or not matching the
                        // terminating sequence
                        for invalid_candidate in invalid_candidates.drain(..) {
                            valid_candidates.set(invalid_candidate, false);
                        }

                        // Return false to keep searching
                        false
                    }
                })
                .ok_or_else(|| {
                    macro_rules! alpha {
                        ($position:ident, $byte:literal) => {
                            alpha_to_byte
                                .0
                                .iter()
                                .copied()
                                .$position(|byte| byte == $byte)
                                .unwrap_or_default() as u8
                        };
                        ($byte:literal) => {
                            alpha!(position, $byte)..=alpha!(rposition, $byte)
                        };
                    }

                    TerminatingSequenceError {
                        octothorpe: alpha!(b'#'),
                        double_quote: alpha!(b'"'),
                    }
                })
        }
    }

    fn try_from_file_paths<P: AsRef<Path>>(
        font_path: P,
        image_path: P,
    ) -> Result<Self, ImageBytesDataError> {
        use ImageBytesDataError as IbdError;

        let font_alpha_data: FontAlphaData =
            FontAlphaData::try_from_file_path(font_path).map_err(IbdError::FontAlphaData)?;
        let gray_image: GrayImage =
            Self::try_get_gray_image(image_path).map_err(IbdError::Image)?;

        Self::try_from_font_alpha_data_and_gray_image(
            &font_alpha_data.char_alpha_grid_arr,
            &font_alpha_data.alpha_to_byte,
            &gray_image,
        )
    }

    fn try_from_font_alpha_data_and_gray_image(
        char_alpha_grid_arr: &CharAlphaGridArr,
        alpha_to_byte: &AlphaToByte,
        gray_image: &GrayImage,
    ) -> Result<Self, ImageBytesDataError> {
        use ImageBytesData as Ibd;

        const DRAWING_BYTE_LEN: usize = CharAlphaGrid::PIXEL_AREA * CharAlphaGridArr::PRINTABLE_LEN;
        const DRAWING_BLOCK_LEN: usize = DRAWING_BYTE_LEN / Ibd::BLOCK_BYTE_WIDTH;
        const DRAWING_AND_PARITY_BYTE_LEN: usize = DRAWING_BYTE_LEN * 2_usize;
        const IMAGE_BYTES_BYTE_LEN: usize = CharAlphaGridArr::padded_byte_len(
            DRAWING_AND_PARITY_BYTE_LEN
                + CharAlphaGridArr::MANIFEST_BIT_LEN / Base64::BITS_PER_BYTE,
        ) + CharAlphaGridArr::NEW_LINE_SLICE_LEN;
        const PADDED_FRAME_BYTE_WIDTH: usize =
            Ibd::FRAME_BYTE_WIDTH + CharAlphaGridArr::NEW_LINE_SLICE_LEN;

        let mut drawing: Vec<u8> = Vec::new();
        let mut parity: Vec<u8> = Vec::new();

        drawing.resize(DRAWING_BYTE_LEN, 0_u8);
        parity.resize(DRAWING_BYTE_LEN, 0_u8);

        for ((drawing_luma, char_alpha), (drawing_dest, parity)) in gray_image
            .pixels()
            .zip(
                char_alpha_grid_arr
                    .0
                    .iter()
                    .flat_map(|char_alpha_grid| char_alpha_grid.0.iter())
                    .copied(),
            )
            .zip(drawing.iter_mut().zip(parity.iter_mut()))
        {
            let drawing: u8 = alpha_to_byte.0[drawing_luma.0[0_usize] as usize];

            *drawing_dest = drawing;
            *parity = CharAlphaGrid::parity(char_alpha, drawing);
        }

        let drawing_byte_width: usize = gray_image.width() as usize;
        let octothorpes: usize =
            Self::try_get_octothorpes(&drawing, drawing_byte_width, alpha_to_byte)
                .map_err(ImageBytesDataError::TerminatingSequence)?;

        let mut manifest: BitVec<u8, Lsb0> =
            BitVec::with_capacity(CharAlphaGridArr::MANIFEST_BIT_LEN);
        let mut image_bytes_data: ImageBytesData = ImageBytesData {
            bytes: Vec::with_capacity(IMAGE_BYTES_BYTE_LEN),
            octothorpes,
        };
        let mut block_len: usize = 0_usize;
        let mut drawing_block_iter: ChunksExact<u8> = drawing.chunks_exact(Self::BLOCK_BYTE_WIDTH);
        let mut parity_block_iter: ChunksExact<u8> = parity.chunks_exact(Self::BLOCK_BYTE_WIDTH);
        let mut push_block = |is_drawing: bool| {
            if block_len % Self::FRAME_BLOCK_WIDTH == 0_usize {
                image_bytes_data
                    .bytes
                    .extend_from_slice(CharAlphaGridArr::NEW_LINE_SLICE);
            }

            for value in if is_drawing {
                drawing_block_iter.next()
            } else {
                parity_block_iter.next()
            }
            .unwrap()
            {
                image_bytes_data.bytes.push(*value);
            }

            block_len += 1_usize;

            manifest.push(is_drawing);

            if manifest.len() % u8::BITS as usize == Base64::BITS_PER_BYTE {
                manifest.push(false);
                manifest.push(false);
            }
        };

        let drawing_block_width: usize = drawing_byte_width / Self::BLOCK_BYTE_WIDTH;
        let side_margin: usize = (Self::FRAME_BLOCK_WIDTH - drawing_block_width) / 2_usize;
        let height: usize = gray_image.height() as usize;
        let top_and_bottom_blocks: usize = DRAWING_BLOCK_LEN - side_margin * height * 2_usize;
        let top_margin: usize = (top_and_bottom_blocks as f32 / Self::FRAME_BLOCK_WIDTH as f32
            * 0.5_f32)
            .ceil() as usize;
        let top_blocks: usize = top_margin * Self::FRAME_BLOCK_WIDTH;

        for _ in 0_usize..top_blocks {
            push_block(false);
        }

        for _ in 0_usize..height {
            for _ in 0_usize..side_margin {
                push_block(false);
            }

            for _ in 0_usize..drawing_block_width {
                push_block(true);
            }

            for _ in 0_usize..side_margin {
                push_block(false);
            }
        }

        for _ in 0_usize..top_and_bottom_blocks - top_blocks {
            push_block(false);
        }

        for manifest_byte in manifest.into_vec().into_iter().map(Base64::encode) {
            if image_bytes_data.bytes.len() % PADDED_FRAME_BYTE_WIDTH == 0_usize {
                image_bytes_data
                    .bytes
                    .extend_from_slice(CharAlphaGridArr::NEW_LINE_SLICE);
            }

            image_bytes_data.bytes.push(manifest_byte);
        }

        image_bytes_data
            .bytes
            .extend_from_slice(CharAlphaGridArr::NEW_LINE_SLICE);

        Ok(image_bytes_data)
    }

    fn octothorpes_string(&self) -> String {
        format!("{0:#<1$}", "", self.octothorpes)
    }

    fn bytes_str(&self) -> &str {
        // SAFETY: The drawing bytes only consist of printable ASCII characters, the parity bytes
        // only consist of bytes that match the regex pattern `[A-Za-z0-9+/]`, and the new line
        // slice is `b" \n"`, which is all valid ASCII, thus valid UTF8-encoded Unicode
        unsafe { from_utf8_unchecked(&self.bytes) }
    }
}

impl Debug for ImageBytesData {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let octothorpes_string: String = self.octothorpes_string();

        write!(
            f,
            "ImageBytesData {{ bytes: br{}\"{}\"{} octothorpes: {} }}",
            octothorpes_string,
            self.bytes_str(),
            octothorpes_string,
            self.octothorpes
        )
    }
}

trait Encode {
    fn encode(value: u8) -> u8;
}

impl Encode for Base64 {
    fn encode(value: u8) -> u8 {
        match value {
            0_u8..=25_u8 => value + b'A',
            26_u8..=51_u8 => value + Self::OFFSET_1,
            52_u8..=61_u8 => value - Self::OFFSET_2,
            62_u8 => b'+',
            63_u8 => b'/',
            _ => {
                eprintln!("unexpected Base64 byte {value}");

                0_u8
            }
        }
    }
}

trait Parity {
    fn parity(alpha: u8, drawing: u8) -> u8;
}

impl Parity for CharAlphaGrid {
    fn parity(alpha: u8, drawing: u8) -> u8 {
        Base64::encode((alpha as f32 / Self::SCALE).round() as u8 ^ drawing & Self::MAX)
    }
}

#[derive(Debug, Parser)]
struct Args {
    #[arg(short, long, default_value_t = FONT_PATH.into())]
    font_path: String,

    #[arg(short, long, default_value_t = IMAGE_PATH.into())]
    image_path: String,

    #[arg(short, long, default_value_t = EXAMPLE_NAME.into())]
    example_name: String,

    #[arg(short, long, default_value_t = TAB_SIZE)]
    tab_size: usize,

    #[arg(short, long, default_value_t)]
    run: bool,
}

fn main() {
    let args: Args = Args::parse();

    if let Err(error) = try_write_example(
        &args.font_path,
        &args.image_path,
        &args.example_name,
        args.tab_size,
    ) {
        eprintln!("Encountered error attempting to write example with args {args:#?}\n{error:#?}");
    } else if args.run {
        Command::new("cargo")
            .args(["run", "--example", &args.example_name, "--", "-h", "10800"])
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .output()
            .ok();
    }
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        lazy_static::lazy_static,
        std::ops::{Div, Mul},
    };

    lazy_static! {
        static ref FONT_ALPHA_DATA: FontAlphaData =
            FontAlphaData::try_from_file_path(FONT_PATH).unwrap();
        static ref GRAY_IMAGE: GrayImage = ImageBytesData::try_get_gray_image(IMAGE_PATH).unwrap();
        static ref IMAGE_BYTES_DATA: ImageBytesData =
            ImageBytesData::try_from_font_alpha_data_and_gray_image(
                &FONT_ALPHA_DATA.char_alpha_grid_arr,
                &FONT_ALPHA_DATA.alpha_to_byte,
                &GRAY_IMAGE
            )
            .unwrap();
        static ref CHAR_ALPHA_GRID_ARR: CharAlphaGridArr =
            CharAlphaGridArr::from_image_bytes(&IMAGE_BYTES_DATA.bytes);
    }

    #[test]
    fn test_font_alpha_data_try_from_file() {
        // Ensure `FONT_ALPHA_DATA` initializes properly, which will panic if it fails
        let _ = &*FONT_ALPHA_DATA;
    }

    #[test]
    fn test_image_bytes_data_try_get_gray_imate() {
        // Ensure `GRAY_IMAGE` initializes properly, which will panic if it fails
        let _ = &*GRAY_IMAGE;
    }

    #[test]
    fn test_image_bytes_data_try_from_font_alpha_data_and_gray_image() {
        // Ensure `IMAGE_BYTES_DATA` initializes properly, which will panic if it fails
        let _ = &*IMAGE_BYTES_DATA;
    }

    #[test]
    fn test_char_alpha_grid_arr_from_image_bytes() {
        // Ensure `CHAR_ALPHA_GRID_ARR` initializes properly, which will panic if it fails
        let _ = &*CHAR_ALPHA_GRID_ARR;
    }

    #[test]
    fn test_cycle_integrity_small() {
        let [
            mut stage_a_6_bits,
            mut stage_b_8_bits,
            mut stage_c_6_bits,
            mut stage_d_8_bits,
        ]: [ByteArray; 4_usize] =
            // SAFETY: This is just a 2D array of bytes, of which `0_u8` is a valid element
            unsafe { MaybeUninit::zeroed().assume_init() };

        fn map_iter<I: Iterator<Item = u8>, F: Fn(f32, f32) -> f32>(
            iter: I,
            dest: &mut ByteArray,
            op: F,
        ) {
            for (input, output) in iter.zip(dest.iter_mut()) {
                *output = op(input as f32, CharAlphaGrid::SCALE).round() as u8
            }
        }

        fn map_array<F: Fn(f32, f32) -> f32>(src: &ByteArray, dest: &mut ByteArray, op: F) {
            map_iter(src.iter().copied(), dest, op)
        }

        map_iter(u8::MIN..=u8::MAX, &mut stage_a_6_bits, f32::div);

        map_array(&stage_a_6_bits, &mut stage_b_8_bits, f32::mul);

        map_array(&stage_b_8_bits, &mut stage_c_6_bits, f32::div);

        assert_eq!(stage_a_6_bits, stage_c_6_bits);

        map_array(&stage_c_6_bits, &mut stage_d_8_bits, f32::mul);

        assert_eq!(stage_b_8_bits, stage_d_8_bits);
    }

    #[test]
    fn test_cycle_integrity_large() {
        let stage_a_6_bits: &ImageBytesData = &IMAGE_BYTES_DATA;
        let stage_b_8_bits: &CharAlphaGridArr = &CHAR_ALPHA_GRID_ARR;
        let stage_c_6_bits: ImageBytesData =
            ImageBytesData::try_from_font_alpha_data_and_gray_image(
                stage_b_8_bits,
                &FONT_ALPHA_DATA.alpha_to_byte,
                &GRAY_IMAGE,
            )
            .unwrap();

        assert_eq!(*stage_a_6_bits, stage_c_6_bits);

        let stage_d_8_bits: CharAlphaGridArr =
            CharAlphaGridArr::from_image_bytes(&stage_c_6_bits.bytes);

        assert!(*stage_b_8_bits == stage_d_8_bits);
    }
}
