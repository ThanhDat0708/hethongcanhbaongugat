const status = document.querySelector('#status');
const connection = document.querySelector('#connection');
const model = document.querySelector('#model');
const dot = document.querySelector('.dot');
const setText = (id, value) => { document.querySelector(id).textContent = value; };
async function refresh() {
  try {
    const response = await fetch('/api/status');
    if (!response.ok) throw new Error('status');
    const data = await response.json();
    connection.textContent = 'CONNECTED';
    model.textContent = data.model_ready ? 'MODEL READY' : 'MODEL OFFLINE';
    status.textContent = data.status;
    setText('#class', data.predicted_class);
    setText('#confidence', data.confidence.toFixed(2));
    setText('#eye-time', `${data.eye_closed_duration.toFixed(2)} s`);
    setText('#mouth-time', `${data.yawn_duration.toFixed(2)} s`);
    setText('#abnormal', `${data.abnormal_for.toFixed(2)} s`);
    setText('#alert', data.alert ? '⚠ CẢNH BÁO TÀI XẾ CÓ DẤU HIỆU NGỦ GẬT!' : '');
    if (data.snapshot) document.querySelector('#snapshot').src = `${data.snapshot}?t=${Date.now()}`;
    dot.style.background = data.status === 'ALERT' ? '#c84c3f' : data.status === 'DROWSINESS' ? '#d28a31' : '#2f755a';
  } catch { connection.textContent = 'OFFLINE'; }
}
refresh(); setInterval(refresh, 500);
