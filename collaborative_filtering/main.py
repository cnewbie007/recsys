import wandb
from collaborative_filtering.preprocessing import CFData
from collaborative_filtering.itemCF import ItemCFRecommender
from collaborative_filtering.userCF import UserCFRecommender

if __name__ == '__main__':
    wandb.init(project='recsys-cf', name='itemcf-vs-usercf')

    data = CFData()

    print('=== ItemCF ===')
    item_cf = ItemCFRecommender()
    item_cf.train(data)
    item_metrics = item_cf.evaluate(data)
    item_cf.save('checkpoints/itemcf')

    print('=== UserCF ===')
    user_cf = UserCFRecommender()
    user_cf.train(data)
    user_metrics = user_cf.evaluate(data)
    user_cf.save('checkpoints/usercf')

    wandb.log({
        'itemcf/recall@10': item_metrics['recall@10'],
        'usercf/recall@10': user_metrics['recall@10'],
    })
    wandb.finish()
