import React, { useState } from 'react';
import axios from 'axios';

const UploadAndDisplayData = () => {
  const [data, setData] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [file1, setFile1] = useState(null);
  const [file2, setFile2] = useState(null);

  const handleFileChange = (event, setFile) => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    const formData = new FormData();
    formData.append('file1', file1);
    formData.append('file2', file2);

    try {
      const response = await axios.post('http://127.0.0.1:5000/api/upload', formData);
      setData(response.data.data[0]);
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error('Error uploading files:', error);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Upload Java Code Files</h1>
      <form onSubmit={handleSubmit} className="mb-4">
        <div className="mb-2">
          <label className="block text-sm font-medium text-gray-700">File 1</label>
          <input
            type="file"
            accept=".java"
            onChange={(e) => handleFileChange(e, setFile1)}
            className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm"
            required
          />
        </div>
        <div className="mb-2">
          <label className="block text-sm font-medium text-gray-700">File 2</label>
          <input
            type="file"
            accept=".java"
            onChange={(e) => handleFileChange(e, setFile2)}
            className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm"
            required
          />
        </div>
        <button
          type="submit"
          className="mt-4 px-4 py-2 bg-blue-500 text-white font-semibold rounded-md shadow-md hover:bg-blue-700"
        >
          Upload and Compare
        </button>
      </form>
      {data && (
        <div className="bg-white shadow-md rounded my-6">
          <table className="min-w-full bg-white">
            <thead className="bg-gray-800 text-white">
              <tr>
                {Object.keys(data).map((key) => (
                  <th key={key} className="w-1/3 text-left py-3 px-4 uppercase font-semibold text-sm">{key}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr>
                {Object.values(data).map((value, index) => (
                  <td key={index} className="text-left py-3 px-4">{value}</td>
                ))}
              </tr>
            </tbody>
          </table>
          <div className="p-4">
            <h2 className="text-xl font-bold">Prediction: {prediction}</h2>
          </div>
        </div>
      )}
    </div>
  );
};

export default UploadAndDisplayData;
