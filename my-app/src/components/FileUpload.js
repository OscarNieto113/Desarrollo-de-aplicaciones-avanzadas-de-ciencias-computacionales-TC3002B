import React, { useState } from 'react';
import axios from 'axios';

const FileUpload = () => {
  const [file1, setFile1] = useState(null);
  const [file2, setFile2] = useState(null);
  const [response, setResponse] = useState(null);

  const handleFileChange1 = (event) => {
    setFile1(event.target.files[0]);
  };

  const handleFileChange2 = (event) => {
    setFile2(event.target.files[0]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    const formData = new FormData();
    formData.append('file1', file1);
    formData.append('file2', file2);

    try {
      const res = await axios.post('http://127.0.0.1:5000', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResponse(res.data);
    } catch (error) {
      console.error('Error uploading files:', error);
    }
  };

  return (
    <div className="file-upload">
      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="file1">File 1:</label>
          <input type="file" id="file1" onChange={handleFileChange1} />
        </div>
        <div>
          <label htmlFor="file2">File 2:</label>
          <input type="file" id="file2" onChange={handleFileChange2} />
        </div>
        <button type="submit">Upload Files</button>
      </form>
      {response && (
        <div className="response">
          <h2>Response:</h2>
          <pre>{JSON.stringify(response, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default FileUpload;
