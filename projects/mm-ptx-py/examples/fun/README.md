<!--
SPDX-FileCopyrightText: 2026 Charles Durham
SPDX-License-Identifier: MIT

MIT License

Copyright (c) 2026 Charles Durham

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->

# MM-PTX Fun Examples

[domain_coloring](https://en.wikipedia.org/wiki/Domain_coloring)

## Domain Coloring
This example uses a PTX Inject kernel that takes two functions meant to be derivatives and plots them by their vector angle and 
magnitude using the HSL colorspace. An example instruction set is provided in Stack PTX. The
instructions are compiled into a cubin and run to create the png stored in the folder. Running the example also
creates a `mp4` video of the plot being animated.

![domain coloring example](domain_coloring/domain_coloring_output.png)

## Domain Coloring Random
This example also uses domain coloring to generating HSL images of functions but instead the functions are generated in `generator_instructions.py`. 64 sets of Stack PTX instructions are generated and each in turn is compiled and ran. The 
output gifs are montaged to assemble the image below:

![domain coloring random example](domain_coloring_random/domain_coloring_output.gif)
