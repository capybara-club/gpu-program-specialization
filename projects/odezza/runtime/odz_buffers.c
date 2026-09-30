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
#include "odz_internal.h"

size_t odz_aligned_bytes(size_t bytes) {
    return bytes>SIZE_MAX-255?SIZE_MAX:(bytes+255)&~(size_t)255;
}

/* Views never own or resize storage. Validate the whole layout before writing
 * any view, so rejection cannot leave partially rebound device inputs. */
int odz_device_views(DeviceBuffer *slab,DeviceBuffer **views,const size_t *sizes,size_t count) {
    size_t i,total=0;
    for(i=0;i<count;i++) {
        size_t n=odz_aligned_bytes(sizes[i]);
        if(n>slab->size-total)return 1;
        total+=n;
    }
    total=0;
    for(i=0;i<count;i++) {
        views[i]->ptr=slab->ptr+total;
        views[i]->size=sizes[i];
        views[i]->borrowed=1;
        total+=odz_aligned_bytes(sizes[i]);
    }
    return 0;
}

int odz_reserve_device_buffers(OdzRuntime *r,size_t trajectories,size_t device,size_t host) {
    if(r->job_epoch||r->pooled_buffers||r->reservation_failed||r->fatal)return 1;
    r->reservation_failed=1;
    if(cuCtxSetCurrent(r->context)||odz_gpu_buffer(r,&r->trajectory_slab,trajectories)||
       odz_gpu_buffer(r,&r->tile_slab,device))return 1;
    r->result_slab=malloc(host);
    if(!r->result_slab) {
        snprintf(r->error,sizeof(r->error),"CPU result-pool reservation failed (%zu bytes)",host);
        return 1;
    }
    r->result_capacity=host;
    r->pooled_buffers=1;
    r->reservation_failed=0;
    return 0;
}
