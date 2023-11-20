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
        print(self.num_users)
        # compute the item-item similarity
        self.similarity_matrix = self.compute_similarity()

    def compute_similarity(self):
        item_item_similarity = cosine_similarity(
            self.user_item_matrix.T,
            dense_output=False
        )
        return item_item_similarity

    def get_item_topk_items(self, item_id):
        # load similarity vector for given item id
        similarities = self.similarity_matrix[item_id]

        # convert to array
        if sp.issparse(similarities):
            similarities = similarities.toarray()
        
        # flatten the similarity vector
        similarities = similarities.flatten()

        # sort the similarity and get the topK indices according to the item id
        top_indices = similarities.argsort()[-(self.topK + 1):]

        return top_indices

    def recommend(self, user_id):
        # get the user id vadality 
        assert user_id < self.num_users, "User does not exist. (User ID out of bound)."

        # get the interation item list of given user
        interated_items = self.user_item_matrix[user_id, :]
        interated_item_indices = interated_items.nonzero()[1]

        # for each interated item, get the list of topK similar items
        rec_list = np.array([], dtype=int)
        print('Number of interated history:', len(interated_item_indices))
        for item_id in interated_item_indices:
            rec_list = np.append(rec_list, self.get_item_topk_items(item_id))

        # remove duplicates and return the list
        interated_item_indices = set(interated_item_indices)
        filtered_list = [item_id for item_id in rec_list if item_id not in interated_item_indices]

        return list(set(filtered_list))
    

if __name__ == '__main__':
    item_cf = UserCollaborativeFiltering(10)
    test_user_ids = [1, 3, 5, 7, 9]
    for user in test_user_ids:
        print()
        print('User ID:', user)
        r_list = item_cf.recommend(user)
        print('Number of recommendations from itemCF:', len(r_list))
        print(r_list)



