# langchain_SelfQueryRetriever.py
"""
SelfQueryRetriever enables the user to query a document with a question and retrieve
the relevant documents. It automatically generates a query from the document and
retrieves the relevant documents. This example presents a simple use case—a hotel
search service with natural language processing capabilities.
"""

import os
from typing import List
from pprint import pprint
from credentials.get_api_credentials import get_api_credentials
from langsmith import Client, traceable
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load the environment variables.
get_api_credentials()
langsmith_client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])

# Assume that the user wants to search for hotels.
documents: List[Document] = [
    Document(
        page_content="The Grand Oceanview Hotel is a 5-star establishment located in Miami, Florida. It features 300 guest rooms, including both suites and standard rooms. The hotel offers an outdoor heated pool, a state-of-the-art gym, multiple dining venues, and a full-service spa. Its conference facilities include several meeting rooms designed for corporate events. The design and amenities are engineered for both business and leisure travelers.",
        metadata={
            "user_rating": 4.9,
            "price_usd_per_night": 309,
            "location": "Miami, Florida",
            "name": "The Grand Oceanview Hotel",
            "website": "https://www.grandoceanviewhotel.com",
        },
    ),
    Document(
        page_content="Downtown City Inn is a budget-friendly hotel in New York City, New York. The property offers 150 rooms equipped with essential amenities such as free Wi-Fi, flat-screen TVs, and private bathrooms. Its central location near major transit hubs and attractions makes it convenient, and the on-site restaurant serves local fare. The service is straightforward, targeting travelers who need basic, reliable accommodations.",
        metadata={
            "user_rating": 3.5,
            "price_usd_per_night": 89,
            "location": "New York City, New York",
            "name": "Downtown City Inn",
            "website": "https://www.downtowncityinn.com",
        },
    ),
    Document(
        page_content="Mountain View Lodge is a 4-star hotel located in Denver, Colorado. The hotel has 200 well-appointed rooms featuring modern decor and standard amenities. Guests can take advantage of an indoor pool, a fitness center with updated equipment, and a restaurant offering regional cuisine. With dedicated conference facilities, this property caters to both business and leisure needs.",
        metadata={
            "user_rating": 4.2,
            "price_usd_per_night": 175,
            "location": "Denver, Colorado",
            "name": "Mountain View Lodge",
            "website": "https://www.mountainviewlodge.com",
        },
    ),
    Document(
        page_content="Lakeside Retreat is a boutique hotel situated in Chicago, Illinois. The property consists of 100 uniquely designed rooms outfitted with contemporary furnishings. It offers an on-site restaurant, a rooftop bar with panoramic views, and a small fitness center. Its proximity to cultural landmarks and the lakefront makes it a practical choice for tourists.",
        metadata={
            "user_rating": 4.0,
            "price_usd_per_night": 220,
            "location": "Chicago, Illinois",
            "name": "Lakeside Retreat",
            "website": "https://www.lakesideretreat.com",
        },
    ),
    Document(
        page_content="Seaside Resort is a 5-star hotel in San Diego, California. With 250 guest rooms including premium ocean-view suites, the resort boasts an outdoor pool, a fully equipped gym, several diverse dining options, and a comprehensive spa. Its location near the beach and popular attractions makes it suitable for discerning vacationers and business visitors alike.",
        metadata={
            "user_rating": 4.8,
            "price_usd_per_night": 350,
            "location": "San Diego, California",
            "name": "Seaside Resort",
            "website": "https://www.seasideresort.com",
        },
    ),
    Document(
        page_content="Urban Comfort Hotel is a mid-range property located in Austin, Texas. It features 180 guest rooms with modern amenities such as high-speed internet, a fitness center, and a complimentary breakfast buffet. Its central location offers quick access to the city’s entertainment and business districts, providing value without unnecessary frills.",
        metadata={
            "user_rating": 4.1,
            "price_usd_per_night": 140,
            "location": "Austin, Texas",
            "name": "Urban Comfort Hotel",
            "website": "https://www.urbancomforthotel.com",
        },
    ),
    Document(
        page_content="Historic Heritage Hotel, situated in Boston, Massachusetts, is a 4-star property that blends classic architecture with modern comforts. It offers 220 guest rooms along with amenities such as complimentary breakfast, in-room dining services, and a compact fitness center. The hotel’s central placement near historical sites and universities makes it an appealing option for culture-minded travelers.",
        metadata={
            "user_rating": 4.3,
            "price_usd_per_night": 210,
            "location": "Boston, Massachusetts",
            "name": "Historic Heritage Hotel",
            "website": "https://www.historicheritagehotel.com",
        },
    ),
    Document(
        page_content="Eco-Friendly Inn in Portland, Oregon, is designed with sustainability in mind. The property provides 120 guest rooms constructed with energy-efficient materials and features eco-conscious amenities such as organic dining options, a green rooftop garden, and a community lounge. Its commitment to sustainability appeals to environmentally aware travelers.",
        metadata={
            "user_rating": 4.6,
            "price_usd_per_night": 160,
            "location": "Portland, Oregon",
            "name": "Eco-Friendly Inn",
            "website": "https://www.ecofriendlyinn.com",
        },
    ),
    Document(
        page_content="City Center Suites is a business-oriented hotel in Atlanta, Georgia. The hotel offers 240 rooms, including several executive suites, and is equipped with a 24-hour business center, multiple meeting rooms, and a quick-service restaurant. Its strategic location in the heart of Atlanta’s business district supports the needs of corporate travelers.",
        metadata={
            "user_rating": 4.4,
            "price_usd_per_night": 195,
            "location": "Atlanta, Georgia",
            "name": "City Center Suites",
            "website": "https://www.citycentersuites.com",
        },
    ),
    Document(
        page_content="Riverside Hotel is a modern establishment in Philadelphia, Pennsylvania. It offers 180 guest rooms that combine contemporary design with essential amenities like high-speed internet, a fitness center, and a dedicated business lounge. Its proximity to major transportation hubs makes it a practical choice for both short stays and longer visits.",
        metadata={
            "user_rating": 4.0,
            "price_usd_per_night": 170,
            "location": "Philadelphia, Pennsylvania",
            "name": "Riverside Hotel",
            "website": "https://www.riversidehotel.com",
        },
    ),
    Document(
        page_content="Sunset Boulevard Hotel is an upscale hotel located in Los Angeles, California. The property includes 260 rooms, many with premium suites offering sweeping views of the city skyline at dusk. It provides extensive amenities, including an outdoor pool, a modern fitness center, several on-site dining options, and versatile event spaces. Its prime location in a bustling urban district serves both leisure and corporate markets.",
        metadata={
            "user_rating": 4.7,
            "price_usd_per_night": 320,
            "location": "Los Angeles, California",
            "name": "Sunset Boulevard Hotel",
            "website": "https://www.sunsetboulevardhotel.com",
        },
    ),
    Document(
        page_content="Coastal Comfort Inn is a family-friendly hotel located in Honolulu, Hawaii. The property offers 150 guest rooms with a focus on modern design and functionality. Amenities include an outdoor pool, a designated kids' play area, and a complimentary breakfast service. Its proximity to the beach and local attractions makes it an ideal choice for family vacations.",
        metadata={
            "user_rating": 4.5,
            "price_usd_per_night": 280,
            "location": "Honolulu, Hawaii",
            "name": "Coastal Comfort Inn",
            "website": "https://www.coastalcomfortinn.com",
        },
    ),
]

# Define the metadata field information.
# Note: We now use "number" instead of "float" for numeric attributes.
metadata_field_info: List[AttributeInfo] = [
    AttributeInfo(
        name="user_rating",
        type="number",
        description="The user rating of the hotel. It's between 0 and 5. Generally, rating more than 3.5 is considered good, more than 4 is considered nice, and more than 4.5 is considered excellent.",
    ),
    AttributeInfo(
        name="price_usd_per_night",
        type="number",
        description="The price per night in USD. Generally, more than 300 USD is considered expensive, and less than 100 USD is considered budget-friendly.",
    ),
    AttributeInfo(
        name="location",
        type="string",
        description="The location of the hotel.",
    ),
    AttributeInfo(
        name="name",
        type="string",
        description="The name of the hotel.",
    ),
    AttributeInfo(
        name="website",
        type="string",
        description="The website of the hotel.",
    ),
]

# Create a vector store from the documents using an embedding model.
vector_store = Chroma.from_documents(
    documents=documents, embedding=OpenAIEmbeddings(model="text-embedding-3-small")
)

# Create an LLM for query understanding.
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create the SelfQueryRetriever using the vector store and metadata field info.
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vector_store,
    document_contents="A summary of available hotels that the guest(user) can book",
    metadata_field_info=metadata_field_info,
)

# Execute a sample query.


@traceable
def ask_to_self_query_retriever(query: str) -> None:
    response = self_query_retriever.invoke(query)
    for document in response:
        print("=" * 100)
        print(f"Hotel name: {document.metadata['name']}")
        print(f"  - description: {document.page_content}")
        print(f"  - user rating: {document.metadata['user_rating']}")
        print(f"  - price per night: ${document.metadata['price_usd_per_night']}")
        print(f"  - location: {document.metadata['location']}")
        print(f"  - website: {document.metadata['website']}")


while True:
    # example question: "Which hotel in Miami has a user rating more than 4 of 5?"
    user_query = input("Enter your query (or 'exit' to quit): ").strip()
    if user_query.lower() == "exit":
        break

    ask_to_self_query_retriever(user_query)
    print("\n\n")
