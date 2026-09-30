/*
 * SPDX-FileCopyrightText: 2026 Charles Durham
 * SPDX-License-Identifier: MIT
 *
 * MIT License
 *
 * Copyright (c) 2026 Charles Durham
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef O_WRITER_H
#define O_WRITER_H

#include "o_sha256.h"

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

typedef struct OWriter {
    char *buffer;
    size_t buffer_size;
    size_t bytes_written;
    OSha256 *sha256;
} OWriter;

static inline OdezzaResult o_writer_init(OWriter *writer, char *buffer, size_t buffer_size, OSha256 *sha256) {
    if (writer == NULL) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    writer->buffer = buffer;
    writer->buffer_size = buffer_size;
    writer->bytes_written = 0u;
    writer->sha256 = sha256;
    return ODEZZA_SUCCESS;
}

static inline OdezzaResult o_writer_write(OWriter *writer, const void *data, size_t data_size) {
    OdezzaResult result;

    if (writer == NULL || (data == NULL && data_size != 0u)) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    if (data_size > SIZE_MAX - writer->bytes_written) {
        return ODEZZA_ERROR_OVERFLOW;
    }
    if (writer->buffer != NULL) {
        if (writer->bytes_written > writer->buffer_size || data_size > writer->buffer_size - writer->bytes_written) {
            return ODEZZA_ERROR_INSUFFICIENT_BUFFER;
        }
        memcpy(writer->buffer + writer->bytes_written, data, data_size);
    }
    if (writer->sha256 != NULL) {
        result = o_sha256_update(writer->sha256, data, data_size);
        if (result != ODEZZA_SUCCESS) {
            return result;
        }
    }
    writer->bytes_written += data_size;
    return ODEZZA_SUCCESS;
}

static inline OdezzaResult o_writer_cstr(OWriter *writer, const char *text) {
    if (text == NULL) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    return o_writer_write(writer, text, strlen(text));
}

static inline OdezzaResult o_writer_format(OWriter *writer, const char *format, ...) {
    char temporary[1024];
    va_list arguments;
    int count;

    if (writer == NULL || format == NULL) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    va_start(arguments, format);
    count = vsnprintf(temporary, sizeof(temporary), format, arguments);
    va_end(arguments);
    if (count < 0) {
        return ODEZZA_ERROR_FORMAT;
    }
    if ((size_t)count >= sizeof(temporary)) {
        return ODEZZA_ERROR_OVERFLOW;
    }
    return o_writer_write(writer, temporary, (size_t)count);
}

static inline OdezzaResult o_writer_bytes(const OWriter *writer, size_t *bytes_written_ret) {
    if (writer == NULL || bytes_written_ret == NULL) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    *bytes_written_ret = writer->bytes_written;
    return ODEZZA_SUCCESS;
}

#endif
