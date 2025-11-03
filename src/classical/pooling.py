def max_pooling(qid_to_ranking, num_docs: int):
    '''Apply max pooling. `qid_to_raking` is a dictionary mapping a qid to a formed ranking, which is a list of tuples (docid, rank, score). Returns another qid to ranking with the pooled result'''
    ans = {}
    for qid, ranking in qid_to_ranking.items():
        scored = set()
        final_ranking = []
        for (docid, _, score) in ranking:
            if docid not in scored:
                final_ranking.append([
                    docid, 
                    len(scored), #also mind the rank
                    score,
                ])

                scored.add(docid)
                if len(scored) == num_docs: break
        ans[qid] = final_ranking
    return ans

def top_k_sum_pooling(qid_to_ranking, num_docs: int, k=3):
    '''Top-k-sum-pooling. `qid_to_raking` is a dictionary mapping a qid to a formed ranking, which is a list of tuples (docid, rank, score). Returns another qid to ranking with the pooled result'''
    ans = {}
    for qid, ranking in qid_to_ranking.items():
        docid_count = {}
        docid_sum = {}

        for (docid, _, score) in ranking:
            if docid_count.get(docid, 0) < k: #Still up until k
                docid_count[docid] = docid_count.get(docid,0) + 1
                docid_sum[docid] = docid_sum.get(docid,0) + score
        
        sorted_sums = sorted(docid_sum.items(), key=lambda item: item[1], reverse=True)
        ans[qid] = [(docid,rank,score) for rank, (docid, score) in enumerate(sorted_sums[:num_docs])]

    return ans