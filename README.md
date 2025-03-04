

### **Tourism Finder – A Smart Travel Recommendation System**  

 

## **Overview**  
In today’s digital world, tourism has transformed with advanced technologies that enhance travel experiences through real-time information, personalized recommendations, and easy navigation. Traditional travel guides and generic recommendation systems often fail to meet individual preferences, making travel planning less efficient.  

**Tourism Finder** is a proximity-based tourism recommendation system that leverages **natural language processing (NLP), machine learning, and geolocation-based filtering** to provide users with highly personalized travel suggestions.  

## **Key Features**  
✅ **Personalized Recommendations** – Content-based filtering ensures that users receive travel suggestions based on their interests.  
✅ **Text Summarization using BERT** – Provides concise and meaningful descriptions of places.  
✅ **Multilingual Support** – Enables users from diverse linguistic backgrounds to explore recommendations.  
✅ **Proximity-Based Filtering** – Uses the **Haversine formula** to compute distances and suggest nearby attractions.  
✅ **Flask-Based Web Application** – Ensures seamless interaction with a user-friendly web interface.  
✅ **Enhanced User Engagement** – Delivers an efficient, dynamic, and accessible travel planning experience.  

## **Technology Stack**  
- **Backend:** Python (Flask)  
- **Frontend:** HTML, CSS, JavaScript  
- **Machine Learning:** BERT (for text summarization)  
- **Database:** SQLite 
- **APIs & Libraries:**  
  - Google Cloud APIs
  - NumPy, Pandas for data processing  
  - Geopy for geolocation calculations  

## **System Architecture**  
1. **User Input:** Travelers provide preferences such as location, interests, and preferred languages.  
2. **Text Summarization:** The system summarizes information using **BERT** for clear and concise travel details.  
3. **Content-Based Filtering:** Matches users with relevant places based on their interests.  
4. **Proximity Calculation:** Uses the **Haversine formula** to rank nearby attractions.  
5. **Multilingual Output:** Results are translated into the user’s preferred language.  
6. **Web Application UI:** A Flask-based interface ensures an intuitive and accessible experience.

    ![image](https://github.com/user-attachments/assets/bdb085bf-251f-4252-84a8-ae6a462eb123)



## **Installation Guide**  
### **Prerequisites:**  
Ensure you have the following installed:  
- Python (>= 3.8)  
- Flask  
- Required dependencies (see `requirements.txt`)  

### **Steps to Run Locally**  
1. Clone the repository:  
   ```sh
   git clone https://github.com/diishanbhag/Tourism_finder.git
   cd Tourism_finder
   ```
2. Create a virtual environment and activate it:  
   ```sh
   python -m venv venv
   source venv/bin/activate  # For Mac/Linux
   venv\Scripts\activate     # For Windows
   ```
3. Install dependencies:  
   ```sh
   pip install -r requirements.txt
   ```
4. Run the application:  
   ```sh
   python app.py
   ```
5. Open your browser and visit:  
   ```
   http://127.0.0.1:5000/
   ```

## **Future Enhancements**  
🔹 Integration with Google Maps API for real-time navigation.  
🔹 User authentication for saving personalized itineraries.  
🔹 AI-powered chatbot for instant travel assistance.  
🔹 Collaborative filtering for community-based recommendations.  

## **Contributions**  
We welcome contributions! Feel free to submit issues, feature requests, or pull requests to improve the system.  

## **License**  
This project is licensed under the [MIT License](LICENSE).  

---

### 🚀 **Start Exploring the World with Tourism Finder!**  
If you find this project useful, don’t forget to ⭐ star the repository!  

Let me know if you want to tweak or add anything! 😊
