// src/App.js
import React from 'react';
import FileUpload from './components/FileUpload';
import './index.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1 className="text-3xl font-bold underline">File Upload</h1>
        <FileUpload />
      </header>
    </div>
  );
}

export default App;
