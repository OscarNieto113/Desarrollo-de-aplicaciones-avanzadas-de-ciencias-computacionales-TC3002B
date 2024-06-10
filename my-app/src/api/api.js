// src/api/api.js
export const fetchData = async () => {
    const response = await fetch('http://127.0.0.1:5000/api/data');
    const data = await response.json();
    return data;
  };
  