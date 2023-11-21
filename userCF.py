import numpy as np
from CF_preprocessing import *
from sklearn.metrics.pairwise import cosine_similarity


class UserCollaborativeFiltering:
    def __init__(self, topK):
        # define top K
        self.topK = topK
        # load the data
        self.user_item_matrix = load_user_item_matrix()
        self.num_users = self.user_item_matrix.shape[0]
        # compute the item-item similarity
        self.similarity_matrix = self.compute_similarity()

    def compute_similarity(self):
        user_user_similarity = cosine_similarity(
            self.user_item_matrix,
            dense_output=False
        )
        return user_user_similarity

    def get_topk_users(self, user_id):
        # load similarity vector for given user id
        similarities = self.similarity_matrix[user_id]

        # convert to array
        if sp.issparse(similarities):
            similarities = similarities.toarray()
        
        # flatten the similarity vector
        similarities = similarities.flatten()

        # sort the similarity and get the topK similar users
        topk_user_indices = similarities.argsort()[-(self.topK + 1):]

        return topk_user_indices[:-1]

    def recommend(self, user_id):
        # check the user id validity 
        assert user_id < self.num_users, "User does not exist. (User ID out of bound)."

        # get the top K similar user indices
        topk_user_indices = self.get_topk_users(user_id)

        # aggregate the top interacted item list for each top K similar user 
        rec_item_list = np.array([], dtype=int)
        for similar_user_id in topk_user_indices:
            interated_items = self.user_item_matrix[similar_user_id, :]
            interated_item_indices = interated_items.nonzero()[1]
            rec_item_list = np.append(rec_item_list, interated_item_indices)

        # get the interacted item indices for the given user
        interated_items = self.user_item_matrix[user_id, :]
        interated_item_indices = interated_items.nonzero()[1]

        # remove duplicates and return the list
        interated_item_indices = set(interated_item_indices)
        filtered_list = [item_id for item_id in rec_item_list if item_id not in interated_item_indices]

        return list(set(filtered_list))
    

if __name__ == '__main__':
    user_cf = UserCollaborativeFiltering(10)
    test_user_ids = [1, 3, 5, 7, 9]
    for user in test_user_ids:
        print()
        print('User ID:', user)
        r_list = user_cf.recommend(user)
        print('Number of recommendations from userCF:', len(r_list))
        print(r_list)
