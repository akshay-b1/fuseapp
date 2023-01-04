import Head from 'next/head'
import Image from 'next/image'
import { Inter } from '@next/font/google'
import styles from '../styles/Home.module.css'
import { useState } from 'react';
import { spawn } from 'child_process';

export default function Home() {
  const [prompt, setPrompt] = useState('');

  const handlePromptChange = (event) => {
    setPrompt(event.target.value);
  }

  const handleButtonClick = () => {
    pythonProcess = spawn('python', ['fuseapp/pages/api/text2image.py', prompt]);
    if (pythonProcess){
      console.log('python process started')
    }
  }

  return (
    <div>
      <input type="text" value={prompt} onChange={handlePromptChange} />
      <button onClick={handleButtonClick}>Run Python File</button>
    </div>
  );
}
