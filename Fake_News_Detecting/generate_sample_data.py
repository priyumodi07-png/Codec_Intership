import pandas as pd
import random
import os

def generate_sample_dataset():
    print("Generating sample dataset for Fake News Detection...")
    
    # Real news samples
    real_titles = [
        "Government announces new economic policy to boost growth",
        "Scientists discover new planet in the habitable zone",
        "Tech giant releases latest smartphone with advanced AI features",
        "Local team wins the national championship in a thrilling match",
        "New study shows the benefits of a balanced diet",
        "Stock market reaches all-time high amid tech rally",
        "City council approves funding for new public park",
        "Breakthrough in renewable energy could lower costs",
        "World leaders meet to discuss climate change initiatives",
        "Health department launches vaccination campaign"
    ]
    
    real_texts = [
        "The government today unveiled a comprehensive economic policy aimed at stimulating growth and reducing unemployment. The policy focuses on infrastructure development and tax incentives for small businesses. Economists believe this is a step in the right direction.",
        "Astronomers have identified an exoplanet that orbits within its star's habitable zone, meaning it could potentially harbor liquid water. The discovery was made using the latest space telescope data.",
        "The new flagship smartphone from the tech giant includes an advanced neural processing unit that powers real-time translation and enhanced computational photography. Reviews have praised its battery life and performance.",
        "In a stunning upset, the local sports team secured the national championship title last night. The final score was 3-2, with the winning goal scored in the last minute of extra time.",
        "A decade-long study involving thousands of participants has concluded that a diet rich in vegetables, lean proteins, and whole grains significantly reduces the risk of cardiovascular disease.",
        "Wall Street saw record gains today as tech stocks surged following positive earnings reports from major industry players. Analysts remain optimistic about the upcoming quarter.",
        "After months of debate, the city council has finalized the budget to build a new 50-acre park downtown, complete with walking trails and a community center. Construction will begin next spring.",
        "Researchers at the National Institute of Technology have developed a new solar cell material that is both cheaper to produce and more efficient than current silicon-based panels.",
        "The summit concluded with a joint declaration by over 50 nations committing to reduce carbon emissions by 30% by the end of the decade through investments in green technology.",
        "In response to the recent outbreak, the department of health has initiated a statewide campaign offering free vaccinations at local clinics and pharmacies."
    ]

    # Fake news samples
    fake_titles = [
        "Aliens land in New York City, demand to see the Mayor",
        "Secret government plot to control minds through tap water revealed",
        "Celebrity clone replaces famous actor on movie set",
        "Eating 10 pounds of chocolate a day cures all diseases",
        "Billionaire plans to tow an iceberg to the desert",
        "Time traveler from 2050 warns of impending doom",
        "Local man discovers infinite energy source in his garage",
        "New law mandates everyone must walk backward on Tuesdays",
        "The moon is actually a giant space station, claims expert",
        "Drinking gasoline gives you superpowers, internet claims"
    ]
    
    fake_texts = [
        "Eyewitnesses in Times Square reported seeing a massive flying saucer descend from the sky. Extraterrestrial beings allegedly emerged and requested an immediate meeting with the city's mayor. Authorities have denied the event.",
        "An anonymous whistleblower has leaked documents suggesting that local municipalities are adding a mind-control chemical to the public water supply. Experts urge citizens to only drink bottled water.",
        "Rumors are swirling that the lead actor in the upcoming blockbuster has been secretly replaced by a genetically engineered clone after walking off set. The studio has yet to comment on these bizarre allegations.",
        "A controversial new 'study' spreading online claims that consuming massive amounts of chocolate daily will make you immune to all known illnesses. Medical professionals strongly advise against this diet.",
        "A eccentric tech billionaire has announced a multi-billion dollar project to lasso an iceberg from Antarctica and tow it to the Sahara desert to create a massive oasis.",
        "A man claiming to be from the year 2050 was arrested yesterday after causing a disturbance. He claimed he was sent back to warn humanity about an uprising of intelligent toaster ovens.",
        "A self-taught inventor says he has built a machine out of household appliances that generates unlimited electricity. However, he refuses to let scientists examine his invention.",
        "A fake news article going viral claims that a newly passed bill will require all citizens to walk backward on Tuesdays to promote 'balance'. No such law exists.",
        "A self-proclaimed astronomer has published a video claiming that the moon is hollow and serves as an alien observation post. The video has garnered millions of views despite being debunked.",
        "A dangerous internet challenge is encouraging teens to ingest small amounts of gasoline, claiming it enhances physical abilities. Doctors warn that this is highly toxic and potentially fatal."
    ]

    data = []
    
    # Generate 150 Real and 150 Fake by mixing and matching, adding slight noise
    for _ in range(150):
        t = random.choice(real_titles)
        txt = random.choice(real_texts)
        data.append({'title': t, 'text': txt, 'label': 'REAL'})
        
        t_fake = random.choice(fake_titles)
        txt_fake = random.choice(fake_texts)
        data.append({'title': t_fake, 'text': txt_fake, 'label': 'FAKE'})

    # Shuffle the dataset
    random.shuffle(data)
    
    df = pd.DataFrame(data)
    
    # ensure directory exists
    os.makedirs('dataset', exist_ok=True)
    df.to_csv('dataset/fake_news_sample.csv', index=False)
    print("Dataset saved to dataset/fake_news_sample.csv with 300 records.")

if __name__ == "__main__":
    generate_sample_dataset()
