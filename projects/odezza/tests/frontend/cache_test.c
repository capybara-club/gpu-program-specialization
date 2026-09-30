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
#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#define _DARWIN_C_SOURCE
#endif
#include "odr_cache.h"
#include <sqlite3.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <unistd.h>
int main(void) {
    char dir[]="/tmp/odezza-sqlite-test-XXXXXX",fresh[]="/tmp/odezza-sqlite-fresh-XXXXXX",error[512],path[1200];
    OdrCache *c=NULL;void *data=NULL;size_t size=0;sqlite3 *db=NULL;
    char *large=malloc(700000);int status,i;pid_t children[4];
    assert(large&&mkdtemp(dir));memset(large,0x35,700000);
    assert(!strcmp(sqlite3_libversion(),"3.53.4"));
    assert(!setenv("ODEZZA_TEMPLATE_CACHE_BYTES","1048576",1));
    /* All children start before any schema exists. */
    assert(mkdtemp(fresh));
    for(i=0;i<4;i++) {
        children[i]=fork();assert(children[i]>=0);
        if(!children[i]) {
            assert(!odr_cache_open(fresh,&c,error,sizeof(error)));
            assert(!odr_cache_lock(c,"fresh"));
            if(odr_cache_get(c,"fresh",&data,&size)==1)assert(!odr_cache_put(c,"fresh","first",5,1));else free(data);
            odr_cache_unlock(c);odr_cache_close(c);_exit(0);
        }
    }
    for(i=0;i<4;i++){assert(waitpid(children[i],&status,0)==children[i]);assert(WIFEXITED(status)&&WEXITSTATUS(status)==0);}
    assert(!odr_cache_open(dir,&c,error,sizeof(error)));
    assert(!odr_cache_lock(c,"shape-a"));
    assert(odr_cache_get(c,"shape-a",&data,&size)==1);
    assert(!odr_cache_put(c,"shape-a","abc",3,.5));
    assert(!odr_cache_get(c,"shape-a",&data,&size)&&size==3&&!memcmp(data,"abc",3));free(data);
    odr_cache_unlock(c);odr_cache_close(c);
    assert(!odr_cache_open(dir,&c,error,sizeof(error)));
    assert(!odr_cache_get(c,"shape-a",&data,&size));free(data);
    snprintf(path,sizeof(path),"%s/templates.sqlite3",dir);assert(sqlite3_open(path,&db)==SQLITE_OK);
    assert(sqlite3_exec(db,"UPDATE artifacts SET artifact=x'616264' WHERE key='shape-a'",NULL,NULL,NULL)==SQLITE_OK);
    sqlite3_close(db);assert(odr_cache_get(c,"shape-a",&data,&size)==1);
    assert(odr_cache_invalidations(c)==1);
    assert(!odr_cache_put(c,"big-a",large,700000,1));assert(!odr_cache_put(c,"big-b",large,700000,1));
    assert(odr_cache_evictions(c)>=1);assert(odr_cache_get(c,"big-a",&data,&size)==1);
    assert(!odr_cache_get(c,"big-b",&data,&size)&&size==700000);free(data);
    assert(odr_cache_put(c,"too-big",large,1048577,1)==1);
    assert(!odr_cache_remove(c,"big-b"));odr_cache_close(c);
    /* Process-safe miss lock: exactly one compiler for a shared key. */
    for(i=0;i<4;i++) {
        children[i]=fork();assert(children[i]>=0);
        if(!children[i]) {
            assert(!odr_cache_open(dir,&c,error,sizeof(error)));
            assert(!odr_cache_lock(c,"concurrent"));
            if(odr_cache_get(c,"concurrent",&data,&size)==1)assert(!odr_cache_put(c,"concurrent","winner",6,1));else free(data);
            odr_cache_unlock(c);odr_cache_close(c);_exit(0);
        }
    }
    for(i=0;i<4;i++){assert(waitpid(children[i],&status,0)==children[i]);assert(WIFEXITED(status)&&WEXITSTATUS(status)==0);}
    assert(!odr_cache_open(dir,&c,error,sizeof(error)));assert(!odr_cache_get(c,"concurrent",&data,&size)&&size==6);free(data);odr_cache_close(c);
    /* Sustained eviction checkpoints instead of accumulating every past blob.
     * The payload budget excludes SQLite overhead and externally pinned WAL
     * snapshots; this checks the cache's normal short-reader behavior. */
    assert(!odr_cache_open(dir,&c,error,sizeof(error)));
    for(i=0;i<200;i++) {
        char key[64];struct stat st;
        snprintf(key,sizeof(key),"soak-%03d",i);
        assert(!odr_cache_put(c,key,large,700000,1));
        snprintf(path,sizeof(path),"%s/templates.sqlite3-wal",dir);
        assert(!stat(path,&st)&&st.st_size<32*1024*1024);
    }
    odr_cache_close(c);
    snprintf(path,sizeof(path),"%s/templates.sqlite3",dir);
    assert(sqlite3_open(path,&db)==SQLITE_OK);
    assert(sqlite3_exec(db,"PRAGMA user_version=99",NULL,NULL,NULL)==SQLITE_OK);sqlite3_close(db);
    assert(odr_cache_open(dir,&c,error,sizeof(error))==-1&&strstr(error,"unsupported"));
    free(large);printf("PASS SQLite %s cold/warm, corruption, eviction soak, oversize, fresh/concurrent initialization, schema: %s\n",sqlite3_libversion(),dir);
    return 0;
}
