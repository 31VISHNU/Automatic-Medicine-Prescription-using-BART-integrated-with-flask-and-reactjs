// MailCard.js
import React, { useState } from 'react';
import './MailCard.css';

const MailCard = ({ title, nameofpatient, ageofpatient, diseaseofpatient, medicineforpatient, heightofpatient, weightofpatient }) => {
  const [isOverlayVisible, setIsOverlayVisible] = useState(false);

  const toggleOverlay = () => {
    setIsOverlayVisible(!isOverlayVisible);
  };

  return (
    <div className={`mail-card ${isOverlayVisible ? 'overlay-visible' : ''}`} onClick={toggleOverlay}>
      <div className="mail-content">
        <h2>{title}</h2>
        <h4>Name:{nameofpatient}</h4>
        <h4>Age:{ageofpatient}</h4>
        <h4>Height:{heightofpatient}</h4>
        <h4>Weight:{weightofpatient}</h4>
        <h4>Disease:{diseaseofpatient}</h4>
        <h4>Medicine:</h4>
        <p>{medicineforpatient}</p>
      </div>
      {isOverlayVisible && <div className="overlay"></div>}
    </div>
  );
};

export default MailCard;
