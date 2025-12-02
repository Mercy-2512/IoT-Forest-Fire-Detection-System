"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"

const RippleEffect = () => {
  const [isFire, setIsFire] = useState(false)
  const [temperature, setTemperature] = useState(30)
  const [humidity, setHumidity] = useState(80)
  const [mq2, setMq2] = useState(2000)
  const [fire_risk, setFireRisk] = useState(0)

  const baseColor = isFire ? "red" : "green"
  const text = isFire ? "Fire Detected" : "No Fire Detected"

  const fetchFireRisk = async () => {
    try {
      const requestData = { temperature, humidity, mq2_value: mq2 };
      console.log("Sending data:", requestData); // Debugging
  
      const response = await fetch("http://127.0.0.1:5000/fire", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestData),
      });
  
      if (!response.ok) {
        // Log full error response
        const errorText = await response.text();
        throw new Error(`Error ${response.status}: ${errorText}`);
      }
  
      const data = await response.json();
      console.log("Response from backend:", data); // Debugging
  
      setIsFire(data.status === "Fire Detected");
      setFireRisk(data.fire_risk);
    } catch (error) {
      console.error("Error fetching fire risk:", error);
    }
  };

  useEffect(() => {
    const interval = setInterval(fetchFireRisk, 5000) // Fetch data every 5 seconds
    return () => clearInterval(interval)
  }, [temperature, humidity, mq2])

  return (
    <div className="relative w-full h-screen flex flex-col items-center justify-center bg-gray-900 overflow-hidden">
      <div className="mb-4 text-white text-xl">
        Temperature: {temperature}°C | Humidity: {humidity}% | MQ2: {mq2}
      </div>
      {["red", "orange", "yellow"].map((color, index) => (
        <motion.div
          key={index}
          className="absolute rounded-full"
          style={{
            background: `radial-gradient(circle, ${baseColor} 0%, rgba(0,0,0,0) 70%)`,
            width: "80vmin",
            height: "80vmin",
          }}
          animate={{ scale: [0.5, 4.5], opacity: [0.7, 0] }}
          transition={{ duration: 4, delay: index * 2, repeat: Infinity, ease: "linear" }}
        />
      ))}
      <div className="z-10 text-4xl font-bold text-white text-center">{text}</div>
      <div className="mb-4 text-white text-xl">
        Fire Risk: {fire_risk}
      </div>
    </div>
  )
}

export default RippleEffect
