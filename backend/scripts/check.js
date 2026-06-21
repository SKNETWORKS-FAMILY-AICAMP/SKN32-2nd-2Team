'use strict';
// 무설치 스모크: 레이어 로드 + 라우트 매칭 + usecase(파일 원천) 동작.
const assert = require('assert');
const { matchRoute } = require('../src/server');
const uc = require('../src/application/usecases');
const { riskLevel, ensemble } = require('../src/domain/rules');
const { validateModelSubmit } = require('../src/validators/schemas');

assert.strictEqual(riskLevel(0.9), 'high');
assert.strictEqual(riskLevel(0.1), 'low');
assert.strictEqual(validateModelSubmit({ model_name: 'm', model_type: 'tree', artifact_path: 'a' }), null);
assert.ok(validateModelSubmit({ model_name: 'm' }));

const r = matchRoute('GET', '/models/CatBoost/charts/roc');
assert.ok(r && r.params.model === 'CatBoost' && r.params.name === 'roc');

const s = uc.getDashboardSummary();
console.log('[check] dashboard summary best:', s.best_model, s.best_auc, '| models:', s.models.length);
const e = ensemble([{ prob: 0.8, weight: 2 }, { prob: 0.4, weight: 1 }]);
assert.ok(Math.abs(e.prob_ensemble - (0.8 * 2 + 0.4) / 3) < 1e-9);

(async () => {
  const sub = await uc.submitModel({ model_name: 'CatBoost_Churn_v2', model_type: 'tree', artifact_path: 'models/churn/cb.cbm', is_active: true, metrics: { roc_auc: 0.791 } });
  assert.ok(sub.model_id, 'submit failed');
  const list = await uc.listModels();
  console.log('[check] submit model_id:', sub.model_id, '| mode:', sub.mode, '| models:', list.models.length);
  console.log('[check] 모든 단언 통과 ✅ (백엔드 레이어·라우트·usecase 정상)');
  process.exit(0);   // mysql 풀이 열려 있으면 프로세스가 안 끝나므로 명시 종료
})();
