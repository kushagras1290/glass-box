import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { config } from './config'; // validated at import time — crashes loudly on bad values
import { log } from './log';

log('info', 'app.boot', { modelId: config.modelId, dtype: config.dtype });

const container = document.getElementById('root');
if (!container) throw new Error('#root missing from index.html');

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
