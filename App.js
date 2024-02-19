
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css'; 
import MailCard from './MailCard';
import blinkingImage from 'C:/Users/vishn/finalyear/project/project/src/Datasets/ai-in-hospitals.png'; 
const App = () => {
  const [name, setName] = useState('');
  const [age, setAge] = useState('');
  const [day, setDay] = useState('');
  const [symptom, setSymptom] = useState('');
  const [result, setResult] = useState([]);
  const [selectedItems, setSelectedItems] = useState([]);
  const [disease, setDisease] = useState('');
  const [med, setMedicine] = useState([]);
  const [showInputForm, setShowInputForm] = useState(false);
  const [weight, setweight] = useState('');
  const [height, setheight] = useState('');
  useEffect(() => {
    const rotateATD = async () => {
      await new Promise(resolve => setTimeout(resolve, 10000));
      setShowInputForm(true);
    };

    rotateATD();
  }, []);

  const handleChat = async () => {
    try {
      const response = await axios.post('http://localhost:5000/process_chat', {
        name: name,
        age: parseInt(age),
        symptoms: symptom,
      });

      setResult(response.data.result);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  const handleCheckboxChange = (item) => {
    setSelectedItems((prevSelectedItems) => {
      if (prevSelectedItems.includes(item)) {
        return prevSelectedItems.filter((selectedItem) => selectedItem !== item);
      } else {
        return [...prevSelectedItems, item];
      }
    });
  };

  const predict = async () => {
    try {
      const response = await axios.post('http://localhost:5000/predict', {
        item: selectedItems,
        day: parseInt(day),
      });
      setDisease(response.data.disease);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  const generateMedicine = async () => {
    try {
      const response = await axios.post('http://localhost:5000/medicine', {
        dis: disease,
      });
      setMedicine(response.data.med);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div className='bb'>
    <div className="container">
      {showInputForm ? (
        <div className="chatbox">
          <h1>ATD</h1>
          <div>
            <label>Enter Name:</label>
            <input type="text" value={name} onChange={(e) => setName(e.target.value)} />
          </div>
          <div>
            <label>Enter Age:</label>
            <input type="number" value={age} onChange={(e) => setAge(e.target.value)} />
          </div>
          <div>
            <label>Enter number of days:</label>
            <input type="number" value={day} onChange={(e) => setDay(e.target.value)} />
          </div>
          <div>
            <label>Enter your weight:</label>
            <input type="number" value={weight} onChange={(e) => setweight(e.target.value)} />
          </div>
          <div>
            <label>Enter your height:</label>
            <input type="number" value={height} onChange={(e) => setheight(e.target.value)} />
          </div>
          <div>
            <label>Enter Symptom:</label>
            <input type="text" value={symptom} onChange={(e) => setSymptom(e.target.value)} />
          </div>
          <button onClick={handleChat}>Chat</button>
          <div className="result-container">
            <label>Select Symptom:</label>
            {result.map((item, index) => (
              <div key={index} className="checkbox-item">
                <input
                  type="checkbox"
                  id={`resultCheckbox${index}`}
                  value={item}
                  onChange={() => handleCheckboxChange(item)}
                />
                <label htmlFor={`resultCheckbox${index}`}>{item}</label>
              </div>
            ))}
            <div>
              <button onClick={predict}>Find my disease</button>
            </div>
          </div>
          <br></br>
          <br></br>
          <div>
            <button onClick={generateMedicine}>Generate my medicine</button>
          </div>
          <div className="medicine-container">
            <MailCard title="Prescription" nameofpatient={name} ageofpatient={age} diseaseofpatient={disease} medicineforpatient={med} heightofpatient={height} weightofpatient={weight} />
          </div>
        </div>
      ) : (
        <img className="blinking-image" src={blinkingImage} alt="Blinking Image" />
      )}
      <div className="chat-bubble">ATD</div>
    </div>
    </div>
  );
};

export default App;
