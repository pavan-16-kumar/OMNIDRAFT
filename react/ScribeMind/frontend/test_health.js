import fetch from 'node-fetch';

async function checkHealth() {
  try {
    const response = await fetch('http://127.0.0.1:8000/health');
    console.log(await response.json());
  } catch (err) {
    console.error(err.message);
  }
}

checkHealth();
