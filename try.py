import random

# Generate user data for 500 users
user_data = {}

for i in range(1, 501):
    user_id = f"user{i}"
    user_data[user_id] = []
    
    # Generate random historical charging data for each user
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        times = random.sample(range(24), random.randint(3, 6))  # Randomly select 3 to 6 charging times per day
        user_data[user_id].append({'day': day, 'times': times})

# Print example of generated user data
for user_id, data in user_data.items():
    print(f"User: {user_id}, Data: {data[:2]}")
