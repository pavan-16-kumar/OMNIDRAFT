const axios = require('axios');
async function test() {
  try {
    const res = await axios.get('http://127.0.0.1:8000/health');
    console.log("Success:", res.data);
  } catch(e) {
    console.error("Failed:", e.message);
  }
}
test();
