import streamlit as st
import chromadb
from google.generativeai import GenerativeModel

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="rag_docs")

# Initialize Google Generative AI model
gen_model = GenerativeModel("gemini-pro")

# Predefined documents
documents = [
    "Operating the Climate Control System  Your Googlecar has a climate control system that allows you to adjust the temperature and airflow in the car. To operate the climate control system, use the buttons and knobs located on the center console.  Temperature: The temperature knob controls the temperature inside the car. Turn the knob clockwise to increase the temperature or counterclockwise to decrease the temperature. Airflow: The airflow knob controls the amount of airflow inside the car. Turn the knob clockwise to increase the airflow or counterclockwise to decrease the airflow. Fan speed: The fan speed knob controls the speed of the fan. Turn the knob clockwise to increase the fan speed or counterclockwise to decrease the fan speed. Mode: The mode button allows you to select the desired mode. The available modes are: Auto: The car will automatically adjust the temperature and airflow to maintain a comfortable level. Cool: The car will blow cool air into the car. Heat: The car will blow warm air into the car. Defrost: The car will blow warm air onto the windshield to defrost it.",
    "Your Googlecar has a large touchscreen display that provides access to a variety of features, including navigation, entertainment, and climate control. To use the touchscreen display, simply touch the desired icon.  For example, you can touch the \"Navigation\" icon to get directions to your destination or touch the \"Music\" icon to play your favorite songs.",
    "Shifting Gears Your Googlecar has an automatic transmission. To shift gears, simply move the shift lever to the desired position.  Park: This position is used when you are parked. The wheels are locked and the car cannot move. Reverse: This position is used to back up. Neutral: This position is used when you are stopped at a light or in traffic. The car is not in gear and will not move unless you press the gas pedal. Drive: This position is used to drive forward. Low: This position is used for driving in snow or other slippery conditions.",
    "Alabama’s capital is Montgomery. Alaska’s capital is Juneau. Arizona’s capital is Phoenix. Arkansas’s capital is Little Rock. California’s capital is Sacramento. Colorado’s capital is Denver. Connecticut’s capital is Hartford. Delaware’s capital is Dover. Florida’s capital is Tallahassee. Georgia’s capital is Atlanta. Hawaii’s capital is Honolulu. Idaho’s capital is Boise. Illinois’s capital is Springfield. Indiana’s capital is Indianapolis. Iowa’s capital is Des Moines. Kansas’s capital is Topeka. Kentucky’s capital is Frankfort. Louisiana’s capital is Baton Rouge. Maine’s capital is Augusta. Maryland’s capital is Annapolis. Massachusetts’s capital is Boston. Michigan’s capital is Lansing. Minnesota’s capital is St. Paul. Mississippi’s capital is Jackson. Missouri’s capital is Jefferson City. Montana’s capital is Helena. Nebraska’s capital is Lincoln. Nevada’s capital is Carson City. New Hampshire’s capital is Concord. New Jersey’s capital is Trenton. New Mexico’s capital is Santa Fe. New York’s capital is Albany. North Carolina’s capital is Raleigh. North Dakota’s capital is Bismarck. Ohio’s capital is Columbus. Oklahoma’s capital is Oklahoma City. Oregon’s capital is Salem. Pennsylvania’s capital is Harrisburg. Rhode Island’s capital is Providence. South Carolina’s capital is Columbia. South Dakota’s capital is Pierre. Tennessee’s capital is Nashville. Texas’s capital is Austin. Utah’s capital is Salt Lake City. Vermont’s capital is Montpelier. Virginia’s capital is Richmond. Washington’s capital is Olympia. West Virginia’s capital is Charleston. Wisconsin’s capital is Madison. Wyoming’s capital is Cheyenne."
]

# Add documents to ChromaDB
for i, doc in enumerate(documents):
    collection.add(documents=[doc], ids=[f"doc_{i}"])

st.title("RAG System with Google Generative AI & ChromaDB")

# User query input
query = st.text_input("Ask a question:")
if query:
    results = collection.query(query_texts=[query], n_results=3)
    retrieved_docs = [doc for doc in results['documents'][0]]
    
    # Generate response using Google Generative AI
    prompt = f"Based on the following documents, answer the query: {query}\n\n" + "\n".join(retrieved_docs)
    response = gen_model.generate_content(prompt)
    
    # Display results
    st.subheader("Retrieved Documents")
    for doc in retrieved_docs:
        st.text(doc)
    
    st.subheader("AI Response")
    st.write(response.text)

st.sidebar.header("Settings")
top_k = st.sidebar.slider("Top-K Retrievals", 1, 10, 3)
