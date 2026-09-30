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
#include "o_sha256.h"
#include <sqlite3.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <unistd.h>

struct OdrCache {
    sqlite3 *db;
    char directory[1024],error[512];
    uint64_t budget,evictions,invalidations;
    int lock_fd;
};
static int fail(OdrCache *c,const char *why) {
    snprintf(c->error,sizeof(c->error),"template cache: %s: %s",why,c->db?sqlite3_errmsg(c->db):strerror(errno));
    return -1;
}
static int sql(OdrCache *c,const char *text) {
    return sqlite3_exec(c->db,text,NULL,NULL,NULL)==SQLITE_OK?0:fail(c,text);
}
static void digest(const void *p,size_t n,unsigned char out[32]) {
    OSha256 s;(void)o_sha256_init(&s);(void)o_sha256_update(&s,p,n);(void)o_sha256_final(&s,out);
}
static int scalar(OdrCache *c,const char *text,sqlite3_int64 *value) {
    sqlite3_stmt *s=NULL;int rc;
    if(sqlite3_prepare_v2(c->db,text,-1,&s,NULL)!=SQLITE_OK)return fail(c,"prepare scalar");
    rc=sqlite3_step(s);if(rc==SQLITE_ROW)*value=sqlite3_column_int64(s,0);
    sqlite3_finalize(s);return rc==SQLITE_ROW?0:fail(c,"read scalar");
}
/* Called within a short write transaction. Payload budget and physical page
 * ceiling are distinct: pages also contain indexes and schema. FULL auto-vacuum
 * reclaims removed pages on commit; WAL is checkpointed after maintenance. */
static int trim(OdrCache *c,uint64_t incoming) {
    sqlite3_int64 total=0;
    if(scalar(c,"SELECT coalesce(sum(length(artifact)),0) FROM artifacts",&total))return -1;
    while((uint64_t)total>c->budget-incoming) {
        sqlite3_int64 oldest=0;
        if(scalar(c,"SELECT length(artifact) FROM artifacts ORDER BY last_used,key LIMIT 1",&oldest)||
           sql(c,"DELETE FROM artifacts WHERE key=(SELECT key FROM artifacts ORDER BY last_used,key LIMIT 1)"))return -1;
        total-=oldest;++c->evictions;
    }
    return 0;
}
int odr_cache_open(const char *directory,OdrCache **out,char *error,size_t cap) {
    OdrCache *c;char path[1200],q[128];sqlite3_int64 version=0,pagesize=0;const char *env;char *end;
    if(!out)return -1;
    *out=NULL;
    if(!directory||!directory[0]||strlen(directory)>=1024) {
        if(error&&cap)snprintf(error,cap,"invalid template cache directory");
        return -1;
    }
    c=calloc(1,sizeof(*c));
    if(!c){if(error&&cap)snprintf(error,cap,"template cache allocation failed");return -1;}
    c->lock_fd=-1;c->budget=UINT64_C(268435456);
    strcpy(c->directory,directory);env=getenv("ODEZZA_TEMPLATE_CACHE_BYTES");
    if(env) {
        errno=0;c->budget=strtoull(env,&end,10);
        if(errno||!env[0]||env[0]=='-'||*end||c->budget<1048576||c->budget>UINT64_C(8589934592)) {
            snprintf(c->error,sizeof(c->error),"ODEZZA_TEMPLATE_CACHE_BYTES must be 1048576..8589934592");goto bad;
        }
    }
    snprintf(path,sizeof(path),"%s/templates.sqlite3",directory);
    if(sqlite3_open_v2(path,&c->db,SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE|SQLITE_OPEN_FULLMUTEX,NULL)!=SQLITE_OK) {fail(c,"open");goto bad;}
    sqlite3_busy_timeout(c->db,30000);
    /* Serialize schema/journal initialization, including the first concurrent
     * opens of an empty database. Never hold this lock during compilation. */
    if(odr_cache_lock(c,"database-initialization"))goto bad;
    if(scalar(c,"PRAGMA user_version",&version))goto bad;
    if(version!=0&&version!=1) {snprintf(c->error,sizeof(c->error),"unsupported template cache schema %lld",(long long)version);goto bad;}
    if(!version&&sql(c,"PRAGMA auto_vacuum=FULL"))goto bad;
    if(sql(c,"PRAGMA journal_mode=WAL")||sql(c,"PRAGMA synchronous=NORMAL")||
       sql(c,"PRAGMA journal_size_limit=16777216")||sql(c,"PRAGMA wal_autocheckpoint=256")||
       sql(c,"BEGIN IMMEDIATE"))goto bad;
    if(sql(c,"CREATE TABLE IF NOT EXISTS artifacts(key TEXT PRIMARY KEY,artifact BLOB NOT NULL,sha256 BLOB NOT NULL CHECK(length(sha256)=32),compile_seconds REAL NOT NULL,created INTEGER NOT NULL,last_used INTEGER NOT NULL,hits INTEGER NOT NULL DEFAULT 0) WITHOUT ROWID")||
       sql(c,"CREATE INDEX IF NOT EXISTS artifacts_lru ON artifacts(last_used)")||
       sql(c,"PRAGMA user_version=1")||trim(c,0)||sql(c,"COMMIT"))goto bad;
    if(scalar(c,"PRAGMA page_size",&pagesize))goto bad;
    /* Allow bounded B-tree/overflow overhead; this also bounds corrupted inserts.
     * A WAL transaction can transiently contain another copy of modified pages. */
    snprintf(q,sizeof(q),"PRAGMA max_page_count=%llu",(unsigned long long)((2*c->budget+16777216)/(uint64_t)pagesize));
    if(sql(c,q))goto bad;
    odr_cache_unlock(c);*out=c;return 0;
bad:
    if(error&&cap)snprintf(error,cap,"%s",c->error);
    odr_cache_close(c);return -1;
}
int odr_cache_lock(OdrCache *c,const char *key) {
    unsigned char d[32];char p[1200];int fd;
    if(!c||!key||c->lock_fd>=0)return -1;
    digest(key,strlen(key),d);snprintf(p,sizeof(p),"%s/compile-%02u.lock",c->directory,d[0]&63u);
    fd=open(p,O_CREAT|O_RDWR,0600);if(fd<0)return fail(c,"open compile lock");
    while(flock(fd,LOCK_EX))if(errno!=EINTR){close(fd);return fail(c,"compile lock");}
    c->lock_fd=fd;return 0;
}
void odr_cache_unlock(OdrCache *c) {
    if(c&&c->lock_fd>=0){(void)flock(c->lock_fd,LOCK_UN);close(c->lock_fd);c->lock_fd=-1;}
}
int odr_cache_remove(OdrCache *c,const char *key) {
    sqlite3_stmt *s=NULL;int rc;
    if(sqlite3_prepare_v2(c->db,"DELETE FROM artifacts WHERE key=?1",-1,&s,NULL)!=SQLITE_OK)return fail(c,"prepare delete");
    rc=sqlite3_bind_text(s,1,key,-1,SQLITE_TRANSIENT);
    if(rc==SQLITE_OK)rc=sqlite3_step(s);
    sqlite3_finalize(s);
    return rc==SQLITE_DONE?0:fail(c,"delete");
}
int odr_cache_get(OdrCache *c,const char *key,void **data,size_t *size) {
    sqlite3_stmt *s=NULL;int rc,n,valid=0;unsigned char d[32];void *copy=NULL;
    *data=NULL;*size=0;
    if(sqlite3_prepare_v2(c->db,"SELECT artifact,sha256 FROM artifacts WHERE key=?1",-1,&s,NULL)!=SQLITE_OK)return fail(c,"prepare lookup");
    rc=sqlite3_bind_text(s,1,key,-1,SQLITE_TRANSIENT);
    if(rc==SQLITE_OK)rc=sqlite3_step(s);
    if(rc==SQLITE_DONE){sqlite3_finalize(s);return 1;}
    if(rc!=SQLITE_ROW){sqlite3_finalize(s);return fail(c,"lookup");}
    n=sqlite3_column_bytes(s,0);
    if(n>0&&(uint64_t)n<=c->budget&&sqlite3_column_bytes(s,1)==32) {
        digest(sqlite3_column_blob(s,0),(size_t)n,d);
        valid=!memcmp(d,sqlite3_column_blob(s,1),32);
        if(valid) {copy=malloc((size_t)n);if(copy)memcpy(copy,sqlite3_column_blob(s,0),(size_t)n);}
    }
    sqlite3_finalize(s);
    if(!valid){++c->invalidations;return odr_cache_remove(c,key)?-1:1;}
    if(!copy){snprintf(c->error,sizeof(c->error),"template cache allocation failed");return -1;}
    /* Once per second per artifact, not once per launch or configuration. */
    if(sqlite3_prepare_v2(c->db,"UPDATE artifacts SET last_used=unixepoch(),hits=hits+1 WHERE key=?1 AND last_used<unixepoch()",-1,&s,NULL)!=SQLITE_OK){free(copy);return fail(c,"prepare touch");}
    rc=sqlite3_bind_text(s,1,key,-1,SQLITE_TRANSIENT);
    if(rc==SQLITE_OK)rc=sqlite3_step(s);
    sqlite3_finalize(s);
    if(rc!=SQLITE_DONE){free(copy);return fail(c,"touch");}
    *data=copy;*size=(size_t)n;return 0;
}
int odr_cache_put(OdrCache *c,const char *key,const void *data,size_t size,double seconds) {
    sqlite3_stmt *s=NULL;unsigned char d[32];int rc;
    if(!data||!size||!key||strlen(key)>1024)return -1;
    if(size>c->budget||size>INT_MAX)return 1;
    digest(data,size,d);
    if(sql(c,"BEGIN IMMEDIATE"))return -1;
    if(odr_cache_remove(c,key)||trim(c,size))goto bad;
    if(sqlite3_prepare_v2(c->db,"INSERT INTO artifacts VALUES(?1,?2,?3,?4,unixepoch(),unixepoch(),0)",-1,&s,NULL)!=SQLITE_OK){fail(c,"prepare insert");goto bad;}
    if(sqlite3_bind_text(s,1,key,-1,SQLITE_TRANSIENT)!=SQLITE_OK||
       sqlite3_bind_blob(s,2,data,(int)size,SQLITE_STATIC)!=SQLITE_OK||
       sqlite3_bind_blob(s,3,d,32,SQLITE_STATIC)!=SQLITE_OK||
       sqlite3_bind_double(s,4,seconds)!=SQLITE_OK){fail(c,"bind insert");goto bad;}
    rc=sqlite3_step(s);sqlite3_finalize(s);s=NULL;
    if(rc!=SQLITE_DONE){fail(c,"insert");goto bad;}
    if(sql(c,"COMMIT"))goto bad;
    (void)sqlite3_wal_checkpoint_v2(c->db,NULL,SQLITE_CHECKPOINT_PASSIVE,NULL,NULL);
    return 0;
bad:
    sqlite3_finalize(s);(void)sqlite3_exec(c->db,"ROLLBACK",NULL,NULL,NULL);return -1;
}
uint64_t odr_cache_evictions(const OdrCache *c){return c?c->evictions:0;}
uint64_t odr_cache_invalidations(const OdrCache *c){return c?c->invalidations:0;}
const char *odr_cache_error(const OdrCache *c){return c?c->error:"invalid template cache";}
void odr_cache_close(OdrCache *c){if(c){odr_cache_unlock(c);if(c->db)sqlite3_close(c->db);free(c);}}
