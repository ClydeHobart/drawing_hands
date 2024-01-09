use {
    ab_glyph::{Font, FontRef, InvalidFont, OutlineCurve},
    ab_glyph_rasterizer::{Point, Rasterizer},
    bitvec::prelude::*,
    clap::Parser,
    content::*,
    image::{imageops::FilterType, open, DynamicImage, GrayImage, ImageError, ImageResult},
    std::{
        fmt::{Debug, Formatter, Result as FmtResult},
        fs::{create_dir_all, read, read_to_string, write},
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
};

const FONT_PATH: &str = "./assets/ubuntu-font-family-0.83/UbuntuMono-R.ttf";
const IMAGE_PATH: &str = "./assets/Drawing_Hands_3x2.png";
const EXAMPLE_NAME: &str = "drawing_hands";

#[derive(Debug, Parser)]
struct Args {
    #[arg(short, long, default_value_t = FONT_PATH.into())]
    font_path: String,

    #[arg(short, long, default_value_t = IMAGE_PATH.into())]
    image_path: String,

    #[arg(short, long, default_value_t = EXAMPLE_NAME.into())]
    example_name: String,

    #[arg(short, long, default_value_t)]
    run: bool,
}

fn main() {
    let args: Args = Args::parse();

    if let Err(error) =
        ImageBytesData::try_write_example(&args.font_path, &args.image_path, &args.example_name)
    {
        eprintln!("Encountered error attempting to write example with args {args:#?}\n{error:#?}");
    } else if args.run {
        Command::new("cargo")
            .args(["run", "--example", &args.example_name])
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .output()
            .ok();
    }
}

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
                char_alpha_grid.0[Cag::PIXEL_WIDTH * Self::invert_y(y as usize) + x as usize] =
                    (alpha * u8::MAX as f32) as u8;
            });

            alpha_sum / Cag::PIXEL_AREA as f32
        } else {
            char_alpha_grid.0.fill(0_u8);

            0.0_f32
        }
    }

    #[inline(always)]
    fn invert_y(y: usize) -> usize {
        Cag::PIXEL_HEIGHT - y - 1_usize
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

#[derive(Debug)]
enum FontAlphaDataError {
    IoError(IoError),
    FaceParsingError(FaceParsingError),
    InvalidFont(InvalidFont),
    MissingGlyph(char),
    MissingHorAdvance(char),

    #[allow(dead_code)]
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
        for code_point in CagArr::PRINTABLE_RANGE.into_iter() {
            let Some(glyph_id) = face.glyph_index(code_point) else { return Err(FadError::MissingGlyph(code_point)); };

            let Some(actual) = face.glyph_hor_advance(glyph_id) else { return Err(FadError::MissingHorAdvance(code_point)); };

            if let Some(expected) = hor_advance.clone() {
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
        let mut rasterizer: Rasterizer = Rasterizer::new(Cag::PIXEL_WIDTH, Cag::PIXEL_HEIGHT);
        let offset: Vector = Vector {
            x: 0.0_f32,
            y: -face.descender() as f32,
        };
        let scale: Vector = Vector {
            x: Cag::PIXEL_WIDTH as f32 / hor_advance.unwrap() as f32,
            y: Cag::PIXEL_HEIGHT as f32 / face.units_per_em() as f32,
        };
        let mut params: CharAlphaGridParams<FontRef> = CharAlphaGridParams {
            font: &font,
            rasterizer: &mut rasterizer,
            offset,
            scale,
            c: Default::default(),
        };

        let mut char_alpha_grid_arr: CagArr = CagArr::default();

        let mut alpha_char_pairs: Vec<(f32, char)> =
            Vec::with_capacity(CagArr::PRINTABLE_LEN - 1_usize);

        for (index, c) in CagArr::PRINTABLE_RANGE.into_iter().enumerate() {
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

#[derive(Debug)]
enum WriteExampleError {
    InvalidExampleChar(char),
    FailedToCreateDir(IoError),
    FailedToReadMain(IoError),
    ContentContainsNonAscii,
    FailedToWriteExample(IoError),
}

#[derive(Debug)]
enum ImageBytesDataError {
    FontAlphaDataError(FontAlphaDataError),
    ImageError(ImageError),
    TerminatingSequenceError(TerminatingSequenceError),
    WriteExampleError(WriteExampleError),
}

impl ImageBytesData {
    const BLOCK_BYTE_WIDTH: usize = CagArr::BLOCK_BYTE_LEN;
    const BLOCK_BYTE_HEIGHT: usize = 16_usize;
    const LONG_BLOCK_LEN: usize = 12_usize;
    const SHORT_BLOCK_LEN: usize = 8_usize;
    const FRAME_BLOCK_WIDTH: usize = CagArr::FRAME_BLOCK_LEN;
    const FRAME_BYTE_WIDTH: usize = CagArr::FRAME_BYTE_LEN;
    // const LINE_BYTE_WIDTH: usize = 448_usize;

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
            .flat_map(|line| {
                (0_usize..drawing_byte_width_minus_two)
                    .into_iter()
                    .map(|start| &line[start..])
            })
            .filter(|candidate| candidate[..2_usize] == *shortest_terminating_sequence)
            .collect();

        if candidates.is_empty() {
            Ok(1_usize)
        } else {
            let mut valid_candidates: BitVec<u32, Lsb0> = bitvec!(u32, Lsb0; 1; candidates.len());
            let mut invalid_candidates: Vec<usize> = Vec::new();

            (3_usize..terminating_sequence.len())
                .into_iter()
                .find(|octothorpes| {
                    let index: RangeTo<usize> = ..*octothorpes;
                    let terminating_sequence: &[u8] = &terminating_sequence[index];

                    if !valid_candidates.iter_ones().any(|candidate_index| {
                        // Check if the candidate at the specified index is still valid
                        if candidates[candidate_index]
                            .get(index.clone())
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
            FontAlphaData::try_from_file_path(font_path).map_err(IbdError::FontAlphaDataError)?;
        let gray_image: GrayImage =
            Self::try_get_gray_image(image_path).map_err(IbdError::ImageError)?;

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

        const DRAWING_BYTE_LEN: usize = Cag::PIXEL_AREA * CagArr::PRINTABLE_LEN;
        const DRAWING_BLOCK_LEN: usize = DRAWING_BYTE_LEN / Ibd::BLOCK_BYTE_WIDTH;
        const DRAWING_AND_PARITY_BYTE_LEN: usize = DRAWING_BYTE_LEN * 2_usize;
        const IMAGE_BYTES_BYTE_LEN: usize = CagArr::padded_byte_len(
            DRAWING_AND_PARITY_BYTE_LEN + CagArr::MANIFEST_BIT_LEN / Base64::BITS_PER_BYTE,
        ) + CagArr::NEW_LINE_SLICE_LEN;
        const PADDED_FRAME_BYTE_WIDTH: usize = Ibd::FRAME_BYTE_WIDTH + CagArr::NEW_LINE_SLICE_LEN;

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
            *parity = Cag::parity(char_alpha, drawing);
        }

        let drawing_byte_width: usize = gray_image.width() as usize;
        let octothorpes: usize =
            Self::try_get_octothorpes(&drawing, drawing_byte_width, alpha_to_byte)
                .map_err(ImageBytesDataError::TerminatingSequenceError)?;

        let mut manifest: BitVec<u8, Lsb0> = BitVec::with_capacity(CagArr::MANIFEST_BIT_LEN);
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
                    .extend_from_slice(CagArr::NEW_LINE_SLICE);
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
                    .extend_from_slice(CagArr::NEW_LINE_SLICE);
            }

            image_bytes_data.bytes.push(manifest_byte);
        }

        image_bytes_data
            .bytes
            .extend_from_slice(CagArr::NEW_LINE_SLICE);

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

    fn write_example(&self, example_name: &str) -> Result<(), WriteExampleError> {
        use WriteExampleError as Error;

        if let Some(c) = example_name
            .chars()
            .find(|c| !(c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '_'))
        {
            return Err(Error::InvalidExampleChar(c));
        }

        const EXAMPLES_PATH: &str = "./examples/";

        create_dir_all(EXAMPLES_PATH).map_err(Error::FailedToCreateDir)?;

        let main: String = read_to_string("./src/main.rs").map_err(Error::FailedToReadMain)?;

        const CONTENT_STR: &str = concat!("mod ", "content ", "{");

        let after_open_bracket: usize = main.find(CONTENT_STR).unwrap() + CONTENT_STR.len();
        let start: usize = main[after_open_bracket..]
            .find(|c: char| !c.is_whitespace())
            .unwrap()
            + after_open_bracket;

        let mut brackets: usize = 1_usize;

        let end: usize = main[start..]
            .find(|c: char| {
                if c == '}' {
                    brackets -= 1_usize;

                    brackets == 0_usize
                } else {
                    brackets += (c == '{') as usize;

                    false
                }
            })
            .unwrap()
            + start;
        let content: &str = &main[start..end];

        if !content.is_ascii() {
            return Err(Error::ContentContainsNonAscii);
        }

        const INPUT_TEXT_PATH: &str = r#"const INPUT_TEXT_PATH: &str = "";"#;
        const OUTPUT_IMAGE_PATH: &str = r#"const OUTPUT_IMAGE_PATH: &str = "";"#;
        const BYTES: &str = r#"const BYTES: &[u8] = br"";"#;

        let input_text_path_pos: usize =
            content.find(INPUT_TEXT_PATH).unwrap() + INPUT_TEXT_PATH.len() - 2_usize;
        let output_image_path_pos: usize =
            content.find(OUTPUT_IMAGE_PATH).unwrap() + OUTPUT_IMAGE_PATH.len() - 2_usize;
        let bytes_start: usize = content.find(BYTES).unwrap() + BYTES.len() - 3_usize;
        let bytes_end: usize = bytes_start + 2_usize;

        let example_path: String = format!("{EXAMPLES_PATH}{example_name}.rs");
        let octothorpes: String = self.octothorpes_string();

        let mut contents: Vec<u8> =
            // Loose calculation, since what actually gets written is subject to change
            Vec::with_capacity(content.len() + self.bytes.len() + 2_usize * self.octothorpes);

        macro_rules! write_contents {
            ( $($str:expr,)* ) => {
                $(
                    contents.extend_from_slice($str.as_ref());
                )*
            };
        }

        write_contents!(
            content[..input_text_path_pos],
            example_path,
            content[input_text_path_pos..output_image_path_pos],
            format!("./output/{example_name}.jpg"),
            content[output_image_path_pos..bytes_start],
            octothorpes,
            "\"",
            self.bytes,
            "\"",
            octothorpes,
            content[bytes_end..],
        );

        write(example_path, contents).map_err(Error::FailedToWriteExample)
    }

    fn try_write_example<P: AsRef<Path>>(
        font_path: P,
        image_path: P,
        example_name: &str,
    ) -> Result<(), ImageBytesDataError> {
        Self::try_from_file_paths(font_path, image_path)?
            .write_example(example_name)
            .map_err(ImageBytesDataError::WriteExampleError)
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

mod content {
    pub use self::{CharAlphaGrid as Cag, CharAlphaGridArr as CagArr};

    use {
        bitvec::{order::Lsb0, vec::BitVec, view::AsBits},
        clap::Parser,
        image::{GrayAlphaImage, ImageError, LumaA, Pixel},
        std::{
            alloc::{alloc_zeroed, handle_alloc_error, Layout},
            fs::{create_dir_all, read_to_string},
            io::Error as IoError,
            mem::MaybeUninit,
            ops::Range,
            path::Path,
            ptr::NonNull,
        },
    };

    const INPUT_TEXT_PATH: &str = "";
    const OUTPUT_IMAGE_PATH: &str = "";
    const BYTES: &[u8] = br"";

    #[derive(Debug, Parser)]
    struct Args {
        #[arg(short, long, default_value_t = INPUT_TEXT_PATH.into())]
        input_text_path: String,

        #[arg(short, long, default_value_t = OUTPUT_IMAGE_PATH.into())]
        output_image_path: String,

        #[arg(short = 'w', long, default_value_t)]
        min_pixel_width: usize,

        #[arg(short = 'm', long, default_value_t)]
        min_pixel_height: usize,

        #[arg(short, long, default_value_t)]
        tab_size: usize,
    }

    unsafe fn alloc_zeroed_non_null<T>() -> NonNull<T> {
        let layout: Layout = Layout::new::<T>();

        if let Some(non_null) = NonNull::new(alloc_zeroed(layout) as *mut T) {
            non_null
        } else {
            handle_alloc_error(layout)
        }
    }

    pub fn alloc_zeroed_box<T>() -> Box<T> {
        // SAFETY: The calls to `alloc_zeroed_non_null` and `Box::from_raw` are safe because
        // the value returned by the former has ownership immediately assumed by the latter:
        // one allocation, one deallocation (when the value returned by this function is
        // dropped)
        unsafe { Box::from_raw(alloc_zeroed_non_null().as_ptr()) }
    }

    pub struct Base64;

    impl Base64 {
        pub const OFFSET_1: u8 = b'a' - 26_u8;
        pub const OFFSET_2: u8 = 52_u8 - b'0';
        pub const BITS_PER_BYTE: usize = 6_usize;

        pub fn decode(value: u8) -> u8 {
            match value {
                b'A'..=b'Z' => value - b'A',
                b'a'..=b'z' => value - Self::OFFSET_1,
                b'0'..=b'9' => value + Self::OFFSET_2,
                b'+' => 62_u8,
                b'/' => 63_u8,
                _ => {
                    eprintln!("unexpected Base64 byte {value}");

                    0_u8
                }
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
        pub const MAX: u8 = ((1_u16 << Self::BITS) - 1_u16) as u8;
        pub const SCALE: f32 = u8::MAX as f32 / Self::MAX as f32;

        pub fn alpha(drawing: u8, parity: u8) -> u8 {
            (((drawing & Self::MAX) ^ Base64::decode(parity)) as f32 * Self::SCALE).round() as u8
        }
    }

    impl Default for CharAlphaGrid {
        fn default() -> Self {
            // SAFETY: `Self` is just an array of `u8`s, for which zeroed bytes are valid
            unsafe { MaybeUninit::zeroed().assume_init() }
        }
    }

    #[derive(PartialEq)]
    pub struct CharAlphaGridArr(pub Box<[CharAlphaGrid; CagArr::PRINTABLE_LEN]>);

    #[derive(Debug)]
    pub enum CharAlphaGridArrPrintError {
        FailedToReadTextFile(IoError),
        NonAsciiChar(char),
        FailedToSaveImageFile(ImageError),
    }

    impl CharAlphaGridArr {
        pub const PRINTABLE_RANGE: Range<char> = ' '..'\x7F';
        pub const PRINTABLE_LEN: usize = Self::PRINTABLE_RANGE.end as usize
            - Self::PRINTABLE_RANGE.start as usize
            // Add one so the information represented is an easier number to work with (96, not 95)
            + 1_usize;
        pub const BLOCK_BYTE_LEN: usize = 32_usize;
        pub const FRAME_BLOCK_LEN: usize = 14_usize;
        pub const FRAME_BYTE_LEN: usize = CagArr::BLOCK_BYTE_LEN * CagArr::FRAME_BLOCK_LEN;
        pub const MANIFEST_BIT_LEN: usize =
            Cag::PIXEL_AREA * CagArr::PRINTABLE_LEN * 2_usize / CagArr::BLOCK_BYTE_LEN;
        pub const NEW_LINE_SLICE: &'static [u8] = b"\n ";
        pub const NEW_LINE_SLICE_LEN: usize = CagArr::NEW_LINE_SLICE.len();

        pub const fn padded_byte_len(byte_len: usize) -> usize {
            byte_len + (byte_len / CagArr::FRAME_BYTE_LEN + 1_usize) * CagArr::NEW_LINE_SLICE_LEN
        }

        pub fn from_image_bytes(bytes: &[u8]) -> Self {
            const MANIFEST_BYTE_LEN: usize =
                CagArr::padded_byte_len(CagArr::MANIFEST_BIT_LEN / Base64::BITS_PER_BYTE);

            let mut char_alpha_grids: Self = Self::default();
            let mut bit_vec: BitVec<u8, Lsb0> = BitVec::with_capacity(Self::MANIFEST_BIT_LEN);

            for manifest_byte in bytes[bytes.len() - MANIFEST_BYTE_LEN..].iter() {
                if !Self::NEW_LINE_SLICE.contains(manifest_byte) {
                    bit_vec.extend_from_bitslice(
                        &[Base64::decode(*manifest_byte)].as_bits::<Lsb0>()
                            [..Base64::BITS_PER_BYTE],
                    );
                }
            }

            let iter_bytes = |block_index| {
                let byte_index: usize = Self::padded_byte_len(block_index * Self::BLOCK_BYTE_LEN);

                bytes[byte_index..byte_index + Self::BLOCK_BYTE_LEN]
                    .iter()
                    .copied()
            };

            for ((drawing, parity), dest) in bit_vec
                .iter_ones()
                .flat_map(iter_bytes)
                .zip(bit_vec.iter_zeros().flat_map(iter_bytes))
                .zip(
                    char_alpha_grids
                        .0
                        .iter_mut()
                        .flat_map(|char_alpha_grid| char_alpha_grid.0.iter_mut()),
                )
            {
                *dest = Cag::alpha(drawing, parity);
            }

            char_alpha_grids
        }

        pub fn print<P: AsRef<Path>>(
            &self,
            text_path: P,
            image_path: P,
            min_pixel_width: usize,
            min_pixel_height: usize,
            tab_size: usize,
        ) -> Result<(), CharAlphaGridArrPrintError> {
            use CharAlphaGridArrPrintError as Error;

            const PIXEL_WIDTH: usize = Cag::PIXEL_WIDTH;
            const PIXEL_HEIGHT: usize = Cag::PIXEL_HEIGHT;
            const BLACK: LumaA<u8> = LumaA([0_u8, u8::MAX]);

            let text: String = read_to_string(text_path).map_err(Error::FailedToReadTextFile)?;
            let (pixel_width, pixel_height): (usize, usize) = text
                .lines()
                .try_fold((0_usize, 0_usize), |(width, height), line| {
                    if let Some(non_ascii_char) = line.chars().find(|c| !c.is_ascii()) {
                        Err(non_ascii_char)
                    } else {
                        Ok((width.max(line.len()), height + 1_usize))
                    }
                })
                .map(|(width, height)| {
                    (
                        (width * PIXEL_WIDTH).max(min_pixel_width),
                        (height * PIXEL_HEIGHT).max(min_pixel_height),
                    )
                })
                .map_err(Error::NonAsciiChar)?;

            let mut image = GrayAlphaImage::new(pixel_width as u32, pixel_height as u32);
            const CHANNEL_COUNT: usize = LumaA::<u8>::CHANNEL_COUNT as usize;

            for pixel in image.pixels_mut() {
                *pixel = BLACK;
            }

            let mut indices: Vec<usize> = Vec::new();

            for (index, line) in text.lines().enumerate() {
                for byte in line.as_bytes().into_iter().copied() {
                    if byte == b'\t' {
                        for _ in 0_usize
                            ..(indices.len() + tab_size) / tab_size * tab_size - indices.len()
                        {
                            indices.push(0_usize);
                        }
                    } else {
                        indices
                            .push(byte.saturating_sub(Self::PRINTABLE_RANGE.start as u8) as usize);
                    }
                }

                let pixel_y_start: usize = index * PIXEL_HEIGHT;

                for y in pixel_y_start..pixel_y_start + PIXEL_HEIGHT {
                    let start: usize = y % PIXEL_HEIGHT * PIXEL_WIDTH;
                    let alpha_range: Range<usize> = start..start + PIXEL_WIDTH;

                    #[inline(always)]
                    fn byte_index(width: usize, x: usize, y: usize) -> usize {
                        (y * width + x) * CHANNEL_COUNT
                    }

                    for (pixel, alpha) in
                        (*image)[byte_index(pixel_width, 0_usize, y)
                            ..byte_index(pixel_width, indices.len() * PIXEL_WIDTH, y)]
                            .chunks_exact_mut(CHANNEL_COUNT)
                            .map(<LumaA<u8> as Pixel>::from_slice_mut)
                            .zip(indices.iter().copied().flat_map(|index| {
                                self.0[index].0[alpha_range.clone()].iter().copied()
                            }))
                    {
                        pixel.blend(&LumaA([u8::MAX, alpha]));
                    }
                }

                indices.clear();
            }

            image.save(image_path).map_err(Error::FailedToSaveImageFile)
        }
    }

    impl Default for CharAlphaGridArr {
        fn default() -> Self {
            Self(alloc_zeroed_box())
        }
    }

    #[allow(dead_code)]
    fn main() {
        let args: Args = Args::parse();

        if args.output_image_path == OUTPUT_IMAGE_PATH {
            create_dir_all("./output").ok();
        }

        if let Err(error) = CagArr::from_image_bytes(BYTES).print(
            &args.input_text_path,
            &args.output_image_path,
            args.min_pixel_width,
            args.min_pixel_height,
            args.tab_size,
        ) {
            eprintln!("Encountered error attempting to print with args {args:#?}\n{error:#?}");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{io::ErrorKind, process::ChildStdout};

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
                *output = op(input as f32, Cag::SCALE).round() as u8
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

    #[allow(dead_code, unused_imports)]
    fn test_2023_01_27() {
        use std::io::{Read, Write};
        use std::process::Output;
        use std::sync::{Arc, Mutex};

        // let mut child = Command::new("nvim")
        //     .arg("./src/main.rs")
        //     // .args([
        //     //     // "-c",
        //     //     // "set t_ti= t_te= nomore",
        //     //     // "-c",
        //     //     // "hi Statement",
        //     //     // "-e",
        //     //     "./src/main.rs",
        //     // ])
        //     .stdin(Stdio::piped())
        //     .stdout(Stdio::inherit())
        //     .spawn()
        //     .unwrap();

        // let mut stdin = child.stdin.take().unwrap();
        // // let mut stdout = child.stdout.take().unwrap();
        // // std::thread::sleep(std::time::Duration::from_millis(500));

        // // let output: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));

        // std::thread::spawn(move || {
        //     std::thread::sleep(std::time::Duration::from_millis(500));
        //     stdin.write_all(b"\x1B:q").ok();
        //     // stdout
        //     //     .read_to_end(&mut output.lock().as_mut().unwrap())
        //     //     .ok();
        // });

        // // std::thread::sleep(std::time::Duration::from_millis(1000));
        // // child.kill().ok();
        // let Output { stdout, stderr, .. } = child.wait_with_output().unwrap();

        let Output { stdout, stderr, .. } = Command::new("nvim")
            .args([
                "-c",
                "%p",
                // "-c",
                // "set t_ti= t_te= nomore",
                // "-c",
                // "hi Statement",
                // "-c",
                // "q",
                "./src/main.rs",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // .stdout(Stdio::inherit())
            // .stderr(Stdio::inherit())
            .output()
            .unwrap();

        let (stdout, stderr) = (
            String::from_utf8(stdout).unwrap(),
            String::from_utf8(stderr).unwrap(),
        );

        // let stdout = &stdout[stdout.rfind('\n').unwrap()..];

        dbg!(&stdout);
        println!("stdout:\n\"\"\"\n{stdout}\n\"\"\"\nstderr:\n\"\"\"\n{stderr}\n\"\"\"\n");
        // println!("stderr:\n\"\"\"\n{stderr}\n\"\"\"\n");
        dbg!(stdout.len(), stderr.len());
    }

    /// This works somewhat!!!
    #[allow(dead_code)]
    fn test_2023_01_28_17_39() {
        use std::{
            io::Write,
            process::{Child, Output},
            thread::sleep,
            time::Duration,
        };

        let mut child: Child = Command::new("vim")
            .arg("src/main.rs")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();

        sleep(Duration::from_millis(500));

        child
            .stdin
            .take()
            .unwrap()
            .write_all(b"\x1B:q!\x00")
            .unwrap();

        let Output { stdout, stderr, .. } = child.wait_with_output().unwrap();
        let (stdout, stderr) = (
            String::from_utf8(stdout).unwrap(),
            String::from_utf8(stderr).unwrap(),
        );

        // println!("stdout:\n\"\"\"\n{stdout}\n\"\"\"\nstderr:\n\"\"\"\n{stderr}\n\"\"\"\n");
        dbg!(stdout.len(), stderr.len(), &stdout, stderr);
        println!("{}", &stdout[..stdout.rfind('\n').unwrap()]);
    }

    /// doesn't work, `File::read` blocks at the end of the file
    #[allow(dead_code)]
    fn test_2023_01_29_15_21() {
        use std::{
            io::{Read, Write},
            process::{Child, ChildStdin, Output},
            str::from_utf8,
            thread::sleep,
            time::Duration,
        };

        let mut child: Child = Command::new("vim")
            .arg("examples/demo_ex.rs")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();

        let mut lines: Vec<String> = Vec::new();
        let mut out: Vec<u8> = Vec::new();
        let mut stdin: ChildStdin = child.stdin.take().unwrap();
        let mut stdout: ChildStdout = child.stdout.take().unwrap();
        let mut append_lines = |lines: &mut Vec<String>| {
            println!("append_lines called");
            out.clear();

            loop {
                let out_len: usize = out.len();

                const READ_LEN: usize = 256_usize;
                const INTERRUPTED_SLEEP_DUR: Duration = Duration::from_millis(1_u64);

                out.resize(out_len + READ_LEN, 0_u8);

                match stdout.read(&mut out[out_len..]) {
                    Ok(0_usize) => {
                        out.truncate(0_usize);
                        println!("read 0, breaking");

                        break;
                    }
                    Ok(read_len) => {
                        out.truncate(out_len + read_len);
                        println!(
                            "read {read_len} ({:?}), looping",
                            // from_utf8(&out[out_len..])
                            out.len()
                        );
                    }
                    Err(error) if error.kind() == ErrorKind::Interrupted => {
                        out.truncate(out_len);
                        println!("interrupted, sleeping");
                        sleep(INTERRUPTED_SLEEP_DUR);
                    }
                    Err(error) => {
                        out.truncate(out_len);
                        println!("error {error:?}, breaking");

                        break;
                    }
                }
            }

            for line in from_utf8(&out).unwrap().lines() {
                lines.push(line.into());
            }
        };

        append_lines(&mut lines);

        loop {
            let lines_len: usize = lines.len();

            dbg!(lines_len);

            stdin.write_all(b"\x1B[B").unwrap();

            sleep(Duration::from_millis(100));
            append_lines(&mut lines);

            if lines.len() == lines_len {
                break;
            }
        }

        child.kill().unwrap();

        let Output { stdout, stderr, .. } = child.wait_with_output().unwrap();
        let (stdout, stderr) = (
            String::from_utf8(stdout).unwrap(),
            String::from_utf8(stderr).unwrap(),
        );

        // println!("stdout:\n\"\"\"\n{stdout}\n\"\"\"\nstderr:\n\"\"\"\n{stderr}\n\"\"\"\n");
        dbg!(stdout.len(), stderr.len(), stdout, stderr, lines.len());

        for (index, line) in lines.into_iter().enumerate() {
            println!("{index:04}: \"{line}\"");
        }
    }

    /// doesn't work, but is iterated on for the next one
    #[allow(dead_code)]
    fn test_2023_01_30_22_08() {
        use std::{
            fs::File,
            io::{Read, Write},
            process::{Child, ChildStdin},
            thread::sleep,
            time::Duration,
        };

        let mut child: Child = Command::new("vim")
            .arg("examples/demo_ex.rs")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut stdin: ChildStdin = child.stdin.take().unwrap();

        for _ in 0_usize
            ..File::open("examples/demo_ex.rs")
                .ok()
                .map_or(0_usize, |file| {
                    file.bytes()
                        .filter_map(Result::ok)
                        .filter(|byte| *byte == b'\n')
                        .count()
                })
        {
            stdin.write_all(b"\x1B[B").ok();
            sleep(Duration::from_millis(30));
        }

        stdin.write_all(b"\x1B:q!\x00").ok();

        let stdout: String = String::from_utf8(
            child
                .wait_with_output()
                .ok()
                .map_or_else(Vec::new, |output| output.stdout),
        )
        .unwrap_or_default();

        println!("{stdout}");
    }

    /// This works! Todo: parametrize the file, catch case where the final read block fails (need
    /// a worker and a watcher thread for this), delete .swp file
    #[allow(dead_code)]
    fn test_2023_01_30_23_05() {
        use std::{
            fs::File,
            io::{Read, Write},
            process::{Child, ChildStdin},
            thread::sleep,
            time::Duration,
        };

        let mut child: Child = Command::new("vim")
            .arg("examples/demo_ex.rs")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut stdin: ChildStdin = child.stdin.take().unwrap();
        let mut stdout: ChildStdout = child.stdout.take().unwrap();

        let (byte_count, line_count): (usize, usize) = File::open("./examples/demo_ex.rs")
            .ok()
            .map(|file| {
                file.bytes().filter_map(Result::ok).fold(
                    (0_usize, 0_usize),
                    |(byte_count, line_count), byte| {
                        (byte_count + 1_usize, line_count + (byte == b'\n') as usize)
                    },
                )
            })
            .unwrap_or_default();

        for _ in 0_usize..line_count {
            stdin.write_all(b"\x1B[B").ok();
            // sleep(Duration::from_micros(1_000));
        }

        let full_block_read_len: usize = (1_usize
            << (usize::BITS - byte_count.leading_zeros()).saturating_sub(1_u32))
        .max(2_usize);

        // This capacity is an underestimate, but it'll save allocations for a few orders of
        // magnitude
        let mut buf: Vec<u8> = Vec::with_capacity(byte_count);
        let mut has_read_full_block: bool = false;

        loop {
            let buf_len: usize = buf.len();
            let read_len: usize = full_block_read_len - (buf_len & (full_block_read_len - 1_usize));

            buf.resize(buf_len + read_len, 0_u8);

            match stdout.read(&mut buf[buf_len..]) {
                Ok(0_usize) => {
                    buf.truncate(buf_len);
                    println!("read 0, breaking");

                    break;
                }
                Ok(read_len) => {
                    buf.truncate(buf_len + read_len);
                    println!("read {read_len} ({:?}), looping", buf.len());

                    if read_len == full_block_read_len {
                        has_read_full_block = true
                    } else if has_read_full_block {
                        println!("  previously read full block, breaking");

                        break;
                    }
                }
                Err(error) if error.kind() == ErrorKind::Interrupted => {
                    buf.truncate(buf_len);
                    println!("interrupted, sleeping");
                    const INTERRUPTED_SLEEP_DUR: Duration = Duration::from_millis(1_u64);
                    sleep(INTERRUPTED_SLEEP_DUR);
                }
                Err(error) => {
                    buf.truncate(buf_len);
                    println!("error {error:?}, breaking");

                    break;
                }
            }
        }

        // sleep(Duration::from_millis(500));
        child.kill().ok();

        let (stdout, stderr) = (buf, Vec::new());
        let (stdout, stderr) = (
            String::from_utf8(stdout).unwrap(),
            String::from_utf8(stderr).unwrap(),
        );

        dbg!(stdout.len(), stderr.len(), &stdout, stderr);
        // println!("{}", &stdout[..stdout.rfind('\n').unwrap()]);
    }

    // Doesn't quite work, but maybe if the input and output alternate, with one thread for each
    #[allow(dead_code)]
    fn test_2023_02_01_22_50(file: &str) {
        use std::{
            fs::File,
            io::{Read, Write},
            process::{Child, ChildStdin},
            sync::{
                atomic::{AtomicU64, Ordering},
                Arc, Mutex,
            },
            thread::{sleep, spawn},
            time::{Duration, Instant},
        };

        let mut child: Child = Command::new("vim")
            .arg(file)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut stdin: ChildStdin = child.stdin.take().unwrap();
        let mut stdout: ChildStdout = child.stdout.take().unwrap();

        let (byte_count, line_count): (usize, usize) = File::open(file)
            .ok()
            .map(|file| {
                file.bytes().filter_map(Result::ok).fold(
                    (0_usize, 0_usize),
                    |(byte_count, line_count), byte| {
                        (byte_count + 1_usize, line_count + (byte == b'\n') as usize)
                    },
                )
            })
            .unwrap_or_default();

        for _ in 0_usize..line_count {
            stdin.write_all(b"\x1B[B").ok();
            sleep(Duration::from_micros(1_000));
        }

        let full_block_read_len: usize = (1_usize
            << (usize::BITS - byte_count.leading_zeros()).saturating_sub(1_u32))
        .max(2_usize);

        dbg!(full_block_read_len);

        let start: Instant = Instant::now();
        let now_nanos = move || (Instant::now() - start).as_nanos() as u64;

        // This capacity is an underestimate, but it'll save allocations for a few orders of
        // magnitude
        struct ThreadData {
            stdout_bytes: Mutex<Vec<u8>>,
            update_time: AtomicU64,
        }

        let thread_data: Arc<ThreadData> = Arc::new(ThreadData {
            stdout_bytes: Mutex::new(Vec::with_capacity(byte_count)),
            update_time: AtomicU64::new(0_u64),
        });

        let worker_thread_data = thread_data.clone();

        let worker = spawn(move || {
            let mut has_read_full_block: bool = false;
            let mut buf: Vec<u8> = Vec::with_capacity(full_block_read_len);

            buf.resize(full_block_read_len, 0_u8);

            loop {
                worker_thread_data
                    .update_time
                    .store(now_nanos(), Ordering::Release);

                match stdout.read(&mut buf[..]) {
                    Ok(0_usize) => {
                        println!("read 0, breaking");

                        break;
                    }
                    Ok(read_len) => {
                        let mut stdout = worker_thread_data.stdout_bytes.lock().unwrap();
                        stdout.extend_from_slice(&buf[..read_len]);
                        println!("read {read_len} ({:?}), looping", stdout.len());

                        if read_len == full_block_read_len {
                            has_read_full_block = true
                        } else if has_read_full_block {
                            println!("  previously read full block, breaking");

                            break;
                        }
                    }
                    Err(error) if error.kind() == ErrorKind::Interrupted => {
                        println!("interrupted, sleeping");

                        const INTERRUPTED_SLEEP_DUR: Duration = Duration::from_millis(1_u64);
                        sleep(INTERRUPTED_SLEEP_DUR);
                    }
                    Err(error) => {
                        println!("error {error:?}, breaking");

                        break;
                    }
                }
            }
        });

        const MAX_UPDATE_DUR: Duration = Duration::from_millis(5000);
        const WATCHER_SLEEP_DUR: Duration = Duration::from_millis(500_u64);

        while !worker.is_finished()
            && Duration::from_nanos(now_nanos() - thread_data.update_time.load(Ordering::Acquire))
                < MAX_UPDATE_DUR
        {
            sleep(WATCHER_SLEEP_DUR);
        }

        let worker_is_finished: bool = worker.is_finished();
        let over_max_update_dur: bool =
            Duration::from_nanos(now_nanos() - thread_data.update_time.load(Ordering::Acquire))
                >= MAX_UPDATE_DUR;

        dbg!(worker_is_finished, over_max_update_dur);

        // Kill the child process to keep that from staying open
        child.kill().ok();

        // Sleep in case killing the child process now finishes the worker
        sleep(Duration::from_millis(500));

        dbg!(worker.is_finished());

        let stdout = std::mem::take(&mut *thread_data.stdout_bytes.lock().unwrap());

        dbg!(stdout.len(), String::from_utf8(stdout).ok());
        // println!("{}", &stdout[..stdout.rfind('\n').unwrap()]);
    }

    fn test_2023_02_01(file: &str) {
        use std::{
            fs::File,
            io::{Read, Write},
            process::{Child, ChildStdin},
            sync::{
                atomic::{AtomicU64, Ordering},
                Arc, Mutex,
            },
            thread::{sleep, spawn},
            time::{Duration, Instant},
        };

        let mut child: Child = Command::new("vim")
            // .args(["-c", "se noru bo=insertmode,esc", "-c", "set!"])
            .args([
                "-c",
                "se noru bo=insertmode,esc,cursor ttyfast ttymouse=xterm",
            ])
            .arg(file)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut stdin: ChildStdin = child.stdin.take().unwrap();
        let mut stdout: ChildStdout = child.stdout.take().unwrap();

        let (byte_count, _line_count): (usize, usize) = File::open(file)
            .ok()
            .map(|file| {
                file.bytes().filter_map(Result::ok).fold(
                    (0_usize, 0_usize),
                    |(byte_count, line_count), byte| {
                        (byte_count + 1_usize, line_count + (byte == b'\n') as usize)
                    },
                )
            })
            .unwrap_or_default();
        let full_block_read_len: usize = (1_usize
            << (usize::BITS - byte_count.leading_zeros()).saturating_sub(1_u32))
        .max(2_usize);

        dbg!(full_block_read_len);

        // stdin.write_all(b"i").ok();
        // sleep(Duration::from_millis(100_u64));
        // stdin.write_all(b"\x1B").ok();

        // Sleep so that the vim can render the initial view buffer
        sleep(Duration::from_millis(500_u64));

        let start: Instant = Instant::now();
        let now_nanos = move || (Instant::now() - start).as_nanos() as u64;

        // This capacity is an underestimate, but it'll save allocations for a few orders of
        // magnitude
        struct ThreadData {
            stdout_bytes: Mutex<Vec<u8>>,
            update_time: AtomicU64,
        }

        let thread_data: Arc<ThreadData> = Arc::new(ThreadData {
            stdout_bytes: Mutex::new(Vec::with_capacity(byte_count)),
            update_time: AtomicU64::new(0_u64),
        });

        let worker_thread_data = thread_data.clone();

        let worker = spawn(move || {
            let mut buf: Vec<u8> = Vec::with_capacity(full_block_read_len);

            buf.resize(full_block_read_len, 0_u8);

            loop {
                match stdout.read(&mut buf[..]) {
                    Ok(0_usize) => {
                        println!("read 0, breaking");

                        break;
                    }
                    Ok(read_len) => {
                        if read_len == 1_usize && buf[0_usize] == b'\x07' {
                            println!("read bell, breaking");

                            break;
                        }

                        let mut stdout = worker_thread_data.stdout_bytes.lock().unwrap();
                        stdout.extend_from_slice(&buf[..read_len]);

                        if read_len <= 4_usize {
                            println!(
                                "read {read_len} ({:?}, {:?}), looping",
                                stdout.len(),
                                &stdout[stdout.len() - read_len..]
                            );
                        } else {
                            println!("read {read_len} ({:?}), looping", stdout.len());
                        }
                    }
                    Err(error) if error.kind() == ErrorKind::Interrupted => {
                        println!("interrupted, sleeping");

                        const INTERRUPTED_SLEEP_DUR: Duration = Duration::from_millis(1_u64);
                        sleep(INTERRUPTED_SLEEP_DUR);
                    }
                    Err(error) => {
                        println!("error {error:?}, breaking");

                        break;
                    }
                }

                worker_thread_data
                    .update_time
                    .store(now_nanos(), Ordering::Release);
            }
        });

        let mut loop_watcher = |send_down: bool| {
            const MAX_UPDATE_DUR: Duration = Duration::from_millis(100_u64);
            const WATCHER_SLEEP_DUR: Duration = Duration::from_millis(200_u64);

            // let mut prev_update_time: u64 = 0_u64;

            loop {
                if worker.is_finished() {
                    break;
                }

                let update_time: u64 = thread_data.update_time.load(Ordering::Acquire);

                if update_time != 0_u64
                    && Duration::from_nanos(now_nanos() - update_time) > MAX_UPDATE_DUR
                {
                    if !send_down
                    /* || update_time == prev_update_time */
                    {
                        break;
                    }

                    // prev_update_time = update_time;
                    stdin.write_all(b"\x1B[B").ok();
                }

                sleep(WATCHER_SLEEP_DUR);
            }
        };

        // First loop the watcher until the worker reaches the end of the initial display
        loop_watcher(false);

        // Then loop the watcher pressing down. The worker should receive a bell eventually,
        // at which point the worker will be finished and the watcher will end its loop
        loop_watcher(true);

        println!(
            "is worker finished before killing child? {}",
            worker.is_finished()
        );

        // Kill the child process to keep that from staying open
        child.kill().ok();

        // Sleep in case killing the child process now finishes the worker
        sleep(Duration::from_millis(500));

        println!(
            "is worker finished after killing child? {}",
            worker.is_finished()
        );

        let stdout = std::mem::take(&mut *thread_data.stdout_bytes.lock().unwrap());

        dbg!(stdout.len(), String::from_utf8(stdout).ok());
        // println!("{}", &stdout[..stdout.rfind('\n').unwrap()]);
    }

    #[test]
    fn test_vim_io() {
        test_2023_02_01("./lines.txt");
    }
}
