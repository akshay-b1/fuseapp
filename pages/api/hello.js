const { Client } = require('pg');

// adding python code
const {spawn} = require('child_process');

export default async function handler(req, res) {
    let python = spawn('python', ['C:/Users/abelh/Downloads/pointe.py']);
    let dataToSend = '';

    for await (const data of python.stdout){
      //console.log(data.toString());
      dataToSend += data.toString()
    }
  return res.status(200).json({ message: dataToSend})
} 