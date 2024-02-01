import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';  // Import Axios
import './App.css';
import intents from './intents.json';
const App = () => {
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const messagesEndRef = useRef(null);

  const handleSend = async (event) => {
    event.preventDefault();
  
    // Add the user's message to the state
    setMessages([...messages, { text: newMessage, sender: 'user' }]);
    setNewMessage('');
  
    try {
      // Make a POST request to the Flask server using Axios
      const response = await axios.post('https://uniquekmitchatbot8.onrender.com/predict', {
        message: newMessage
      });
      console.log(response.data);

      if (!response.status === 200) {
        throw new Error('Network response was not ok');
      }
  
      const responseData = response.data;
      const botResponseTag = responseData.tag;
      console.log(botResponseTag)
      // Find the corresponding response in your JSON file
      const matchingIntent = intents.intents.find(intent => intent.tag === botResponseTag);
      const botResponse = matchingIntent ? matchingIntent.responses[0] : "I'm sorry, I don't understand that.";
      const botLink = matchingIntent ? matchingIntent.link:null;
      console.log(botResponse)
      setTimeout(() => {
        setMessages((prevMessages) => [
          ...prevMessages,
          { text: botResponse, sender: 'bot', link: botLink},
        ]);
      }, 1000);
    } catch (error) {
      console.error('Error sending message:', error);
    }
  };
  

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  return (
    <div className="chatroom">
      <div className="header">Sarathi</div>
      <ul className="messages">
        {messages.map((message, index) => (
          <li key={index} className={message.sender}>
            {message.text}
            {message.link ? <a href={message.link}>Click here</a> : null}
          </li>
        ))}
        <div ref={messagesEndRef} />
      </ul>
      <form onSubmit={handleSend}>
        <input
          value={newMessage}
          onChange={(e) => setNewMessage(e.target.value)}
          placeholder="Type your message..."
        />
        <button type="submit">Send</button>
      </form>
    </div>
  );
};

export default App;
