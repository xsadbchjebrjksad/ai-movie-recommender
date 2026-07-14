import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
st.set_page_config(page_title="AI Movie Recommender", page_icon="🎬", layout="centered")
 
# ----------------------------
# 1. Built-in movie dataset (title + short plot description)
# ----------------------------
@st.cache_resource
def load_data_and_model():
    movies = pd.DataFrame({
        "title": [
            "Inception", "Interstellar", "The Dark Knight", "The Matrix",
            "Titanic", "The Notebook", "La La Land", "Pride and Prejudice",
            "The Avengers", "Iron Man", "Spider-Man: Into the Spider-Verse", "Thor",
            "Toy Story", "Finding Nemo", "Shrek", "Coco",
            "The Conjuring", "Get Out", "A Quiet Place", "Hereditary",
            "The Hangover", "Superbad", "Step Brothers", "Bridesmaids",
            "The Social Network", "Moneyball", "The Imitation Game", "Steve Jobs",
        ],
        "plot": [
            "A thief who steals corporate secrets through dream-sharing technology is given a chance to have his criminal history erased by planting an idea into a target's subconscious",
            "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival amid a dying earth and time distortion",
            "Batman raises the stakes in his war on crime facing a criminal mastermind known as the Joker who wants to plunge Gotham into anarchy",
            "A computer hacker learns about the true nature of his reality and his role in the war against controlling machines and artificial intelligence",
            "A poor artist and a wealthy young woman fall in love aboard the ill fated ship on its maiden voyage across the ocean",
            "A poor young man and a rich young woman fall deeply in love but are kept apart by class differences and family expectations",
            "A jazz musician and an aspiring actress fall in love in Los Angeles while pursuing their dreams and creative careers",
            "A young woman navigates issues of manners, marriage, morality and misunderstandings in early 19th century England society",
            "Earth's mightiest heroes must come together to stop a mischievous god and his alien army from enslaving humanity",
            "A billionaire industrialist and genius inventor builds a powered exoskeleton suit to fight crime and evil as a superhero",
            "A teenager from Brooklyn becomes the spider hero of his universe and must team up with other spider heroes from parallel dimensions",
            "The powerful thunder god is cast out of Asgard and finds himself exiled on Earth where he must learn humility to become a hero again",
            "A cowboy toy is threatened when a new spaceman toy supplants him as top toy in a boy's room and they must learn to work together",
            "A clownfish father searches the ocean for his missing son with the help of a forgetful blue tang fish they meet along the way",
            "An ogre who values his solitude finds his swamp overrun by fairy tale creatures and sets out to reclaim his land from a scheming lord",
            "A boy is accidentally transported to the land of the dead where he seeks the help of his deceased family to return home",
            "A family experiences terrifying supernatural occurrences in their farmhouse and calls upon paranormal investigators for help",
            "A young African American man visits his white girlfriend's family estate and uncovers a disturbing and horrifying secret",
            "A family must live in silence while hiding from creatures that hunt anything that makes a sound in a post apocalyptic world",
            "A family is haunted by tragic and disturbing occurrences after the death of their secretive grandmother reveals dark family history",
            "A group of friends wake up from a wild bachelor party in Las Vegas with no memory of the previous night and a missing groom",
            "Two co dependent high school seniors are forced to deal with separation anxiety after their plan to stop drinking and party goes wrong",
            "Two middle aged men still living with their parents become instant rivals when their parents fall in love and get married",
            "A maid of honor's life unravels as she plans her best friend's wedding while competing with a new bridesmaid for control",
            "The story of the founders of Facebook and the lawsuits that followed after the social networking site became a global phenomenon",
            "The story of an unconventional baseball manager who uses statistical analysis to build a winning team on a limited budget",
            "A brilliant mathematician leads a team of codebreakers who crack Nazi encryption machines during the second world war",
            "The story of the innovative entrepreneur and his journey building the personal computer company he founded",
        ],
    })
 
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(movies["plot"])
    similarity_matrix = cosine_similarity(tfidf_matrix)
 
    return movies, similarity_matrix
 
 
movies, similarity_matrix = load_data_and_model()
 
 
def recommend(title, top_n=5):
    idx = movies[movies["title"] == title].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:top_n]
    return [(movies.iloc[i]["title"], round(score * 100, 1)) for i, score in scores]
 
 
# ----------------------------
# 2. UI
# ----------------------------
st.title("🎬 AI Movie Recommender")
st.write(
    "A content-based recommendation engine. It reads each movie's plot, "
    "converts it into a TF-IDF vector, and uses **cosine similarity** to find "
    "movies with the most similar themes and storylines."
)
 
st.divider()
 
selected_movie = st.selectbox("Pick a movie you like:", movies["title"].tolist())
 
top_n = st.slider("How many recommendations?", 3, 10, 5)
 
if st.button("🎥 Get Recommendations", type="primary"):
    results = recommend(selected_movie, top_n)
 
    st.subheader(f"Because you liked '{selected_movie}':")
    for title, score in results:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{title}**")
        with col2:
            st.write(f"{score}% match")
        st.progress(min(int(score), 100))
 
st.divider()
st.caption("Built with Streamlit + scikit-learn (TF-IDF + Cosine Similarity)")
