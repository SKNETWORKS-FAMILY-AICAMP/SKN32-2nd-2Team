'use strict';
// infrastructure/mysql — 운영 DB 풀 + repository. MySQL 미설정/미설치 시 메모리 폴백(데모).
const cfg = require('../../config');
let _pool = null, _mode = 'memory';
const mem = { models: [], evals: [], predictions: [] };

function pool() {
  if (_pool || !cfg.mysql) return _pool;
  try { _pool = require('mysql2/promise').createPool(cfg.mysql); _mode = 'mysql'; }
  catch { _pool = null; }
  return _pool;
}
const mode = () => (cfg.mysql && pool() ? 'mysql' : 'memory');

async function q(sql, params = []) {
  const p = pool(); if (!p) return null;
  const [rows] = await p.execute(sql, params); return rows;
}

// ---- repository (19-1: model/evaluation/prediction/dashboard) ----
const modelRepository = {
  async upsert(m) {
    if (mode() === 'mysql') {
      const r = await q(`INSERT INTO model_registry(model_name,model_type,feature_schema_version,label_name,horizon_days,preprocessing_config,dataset_path,artifact_path,metrics_json,is_active)
        VALUES(?,?,?,?,?,?,?,?,?,?) ON DUPLICATE KEY UPDATE model_type=VALUES(model_type),preprocessing_config=VALUES(preprocessing_config),
        dataset_path=VALUES(dataset_path),artifact_path=VALUES(artifact_path),metrics_json=VALUES(metrics_json),is_active=VALUES(is_active)`,
        [m.model_name, m.model_type, m.feature_schema_version || 'v2', m.label_name || 'churn', m.horizon_days || 7,
         JSON.stringify(m.preprocessing_config || {}), m.dataset_path || null, m.artifact_path,
         JSON.stringify(m.metrics || {}), m.is_active ? 1 : 0]);
      let id = r.insertId;
      if (!id) { const rows = await q('SELECT model_id FROM model_registry WHERE model_name=?', [m.model_name]); id = rows && rows[0] ? rows[0].model_id : null; }
      if (m.is_active) await q('UPDATE model_registry SET is_active=0 WHERE model_type=? AND model_name<>?', [m.model_type, m.model_name]);
      return { model_id: id, mode: 'mysql' };
    }
    const id = mem.models.length + 1;
    if (m.is_active) mem.models.forEach((x) => { if (x.model_type === m.model_type) x.is_active = 0; });
    mem.models.push({ model_id: id, ...m, is_active: m.is_active ? 1 : 0 });
    return { model_id: id, mode: 'memory' };
  },
  async list() { return mode() === 'mysql' ? await q('SELECT * FROM model_registry ORDER BY model_id DESC') : mem.models; },
  async active() { return mode() === 'mysql' ? await q('SELECT * FROM model_registry WHERE is_active=1') : mem.models.filter((x) => x.is_active); },
};

const evaluationRepository = {
  async insert(model_id, e) {
    if (mode() === 'mysql') {
      const r = await q(`INSERT INTO model_evaluation(model_id,dataset_tag,split_name,roc_auc,pr_auc,best_threshold,best_f1,eval_predictions_path,shap_summary_path)
        VALUES(?,?,?,?,?,?,?,?,?)`, [model_id, e.dataset_tag || 'churn', e.split_name || 'test', e.roc_auc ?? null, e.pr_auc ?? null,
        e.best_threshold ?? null, e.best_f1 ?? null, e.eval_predictions_path || null, e.shap_summary_path || null]);
      return { eval_id: r.insertId };
    }
    const id = mem.evals.length + 1; mem.evals.push({ eval_id: id, model_id, ...e }); return { eval_id: id };
  },
};

const predictionRepository = {
  async log(p) {
    if (mode() === 'mysql') {
      await q('INSERT INTO prediction_log(model_id,user_id,churn_probability,risk_level,horizon_days,recommended_action) VALUES(?,?,?,?,?,?)',
        [p.model_id || null, p.user_id, p.churn_probability, p.risk_level, p.horizon_days || 7, p.recommended_action || null]);
    } else mem.predictions.push(p);
  },
  async latest(user_id) {
    if (mode() === 'mysql') return (await q('SELECT * FROM prediction_log WHERE user_id=? ORDER BY prediction_id DESC LIMIT 1', [user_id]))[0] || null;
    return [...mem.predictions].reverse().find((x) => x.user_id === user_id) || null;
  },
};

module.exports = { mode, modelRepository, evaluationRepository, predictionRepository, _mem: mem };
