// src/App.js
import React from 'react';
import FileUpload from './components/FileUpload';
import UploadAndDisplayData from './components/UploadAndDisplayData ';
import './index.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <UploadAndDisplayData />
      </header>
    </div>
  );
}

export default App;
