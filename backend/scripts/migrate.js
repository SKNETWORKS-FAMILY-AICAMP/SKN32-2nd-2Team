'use strict';
// 운영 DB(MySQL) 스키마 적용: db/schema_mysql.sql 을 .env의 MYSQL_*로 실행. (project2db 기존 DB 직접 사용)
const fs = require('fs');
const path = require('path');
const cfg = require('../src/config');

(async () => {
  if (!cfg.mysql) { console.error('[migrate] MYSQL_* 미설정(.env) — 중단'); process.exit(1); }
  let mysql;
  try { mysql = require('mysql2/promise'); } catch { console.error('[migrate] npm i mysql2 필요'); process.exit(1); }
  const raw = fs.readFileSync(path.join(__dirname, '..', 'db', 'schema_mysql.sql'), 'utf-8');
  // 주석 줄(--)을 먼저 제거(문 앞 주석 때문에 통째로 버려지던 버그 방지)
  const sql = raw.split('\n').filter((l) => !l.trim().startsWith('--')).join('\n');
  const stmts = sql.split(/;\s*\n/).map((s) => s.trim()).filter((s) => s.length > 0);
  const conn = await mysql.createConnection({ ...cfg.mysql, multipleStatements: false });
  let ok = 0;
  for (const s of stmts) {
    try { await conn.query(s); ok++; }
    catch (e) { console.warn('[migrate] skip:', String(e.message).slice(0, 80)); }
  }
  const [t] = await conn.query('SHOW TABLES');
  await conn.end();
  console.log(`[migrate] 실행 ${ok}/${stmts.length} 문 · 현재 테이블 ${t.length}개 in ${cfg.mysql.database}`);
})();
