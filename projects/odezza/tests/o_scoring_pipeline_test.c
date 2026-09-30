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
#include "odezza.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    OdezzaScoringPipelineCreateInfo create_info;
    OdezzaScoringPipeline *pipeline = NULL;
    OdezzaResult result;
    char error[256];
    size_t error_size = 0u;
    size_t workspace_size = 0u;
    size_t workspace_alignment = 0u;

    memset(&create_info, 0, sizeof(create_info));
    result = odezza_scoring_pipeline_create(&create_info, &pipeline);
    if (result != ODEZZA_ERROR_INVALID_ARGUMENT || pipeline == NULL) {
        fprintf(stderr, "invalid creation did not retain a diagnostic handle\n");
        return 1;
    }
    if (odezza_scoring_pipeline_write_error(pipeline, error, sizeof(error), &error_size) != ODEZZA_SUCCESS || error_size == 0u ||
        strstr(error, "invalid") == NULL
    ) {
        fprintf(stderr, "failed handle did not retain its error message\n");
        return 1;
    }
    if (odezza_scoring_pipeline_workspace_requirements(pipeline, &workspace_size, &workspace_alignment) != ODEZZA_ERROR_BUSY) {
        fprintf(stderr, "failed handle exposed steady-state workspace\n");
        return 1;
    }
    if (odezza_scoring_pipeline_destroy(pipeline) != ODEZZA_SUCCESS) {
        fprintf(stderr, "failed diagnostic handle could not be destroyed\n");
        return 1;
    }
    puts("public scoring handle failure and diagnostic contract: verified");
    return 0;
}
