# `drawing_hands`

## Installation

TODO

## Execution

TODO

## Background

I wanted to create a piece of art that could, to some extent, encapsulate the beauty that I see in code.

I drew inspiration from a few works, including

* [Andy Sloane's "Donut math: how donut.c works"](https://www.a1k0n.net/2011/07/20/donut-math.html)
* [Yusuke Endoh's "Quine Relay"](https://github.com/mame/quine-relay)
* [M.C. Escher's "Drawing Hands"](https://en.wikipedia.org/wiki/Drawing_Hands), the namesake of this project.

## Explanation

This project consists of three programs, `main.rs`, `content.rs`, and `drawing_hands.rs`

### `main.rs`

`main` takes an input font `.ttf` file, an input image, and an output example name. Using the font data and the input image, `main` constructs a large `&[u8]` literal that encodes alpha bitmaps for printable ASCII characters, with a large ASCII-art representation of the input image in the center. It then combines this literal with the existing contents of `content.rs` to make the third program, which will carry the example name (`drawing_hands` by default). Inserted at the top of this third program is a signature of the project, produced by `FIGlet`.

### `content.rs`

`content` takes an input text path and an output image path. `content` opens up the input text file in `vim` to construct a list of (text, color) ordered pairs. By opening up the file itself and keeping track of whitespace characters, `content` also constructs a list of (text, position) ordered pairs. Due to some `vim` shenanigans, "zippering" these two lists is more difficult than one might think, but eventually content has a list of (text, color, position) ordered trios. With these, and the font alpha bitmaps decoded from the literal, `content` then outputs an image of the text file, colored as it would be in the user's `vim` settings. The key piece missing here is that if `content` tries to decode *its own* literal, it fails, as it's literal definition is just `const BYTES: &[u8] = br"";`. That's where the third program, `drawing_hands.rs`, comes in.

### `drawing_hands.rs`

With all the functional code of `content.rs`, albeit it condensed into a much less readable format, plus a valid `BYTES` literal, `drawing_hands` is actually capable of producing the image output that `content` wishes it could.

## Shortcomings

* testing?

TODO