def mini_edit_distance(str1: str, str2: str, cost: list):
    """
    str1: string 1 for edit
    str2: string 2 for edit
    cost: cost for insertion, deletion and substition
    """
    len1 = len(str1)
    len2 = len(str2)
    d = []
    
    for i in range(len1 + 1):
        d.append([None] * (len2 + 1))
        d[i][0] = i
        if i == 0:
            for j in range(len2 + 1):
                d[i][j] = j
    
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            min_i_d = min(d[i][j - 1] + cost[0], d[i - 1][j] + cost[1])
            min_s = d[i - 1][j - 1]
            if str1[i - 1] != str2[j - 1]:
                min_s += cost[2]
            d[i][j] = min(min_i_d, min_s)
    
    for i in range(len(d)):
        print(d[i])

    operation = backtrace(d)
    word1, word2 = [], []
    str1_idx, str2_idx = 0, 0
    op_name = []
    for i in range(len(operation)):
        # nothing to do
        if operation[i] == 0:
            word1.append(str1[str1_idx])
            str1_idx += 1
            word2.append(str2[str2_idx])
            str2_idx += 1
            op_name.append('equal')
        # insertion
        elif operation[i] == 1:
            word1.append('*')
            word2.append(str2[str2_idx])
            str2_idx += 1
            op_name.append('insertion')
        # deletion
        elif operation[i] == 2:
            str1_idx += 1
            op_name.append('deletion')
        # substition
        else:
            word1.append(str1[str1_idx])
            str1_idx += 1
            word2.append(str2[str2_idx])
            str2_idx += 1
            op_name.append('substition')
    
    return d[len1][len2], op_name, (word1, word2)

def backtrace(d):
    """
    d: distance matrix
    operation: 0 symbolize equal character, 1 denotes insertion
               2 denotes deletion, 3 denotes substition
    """
    operation = []
    i = len(d) - 1
    j = len(d[0]) - 1
    
    while i >= 0 or j >= 0:
        # insertion
        if j > 0 and d[i][j] > d[i][j - 1]:
            operation.append(1)
            j -= 1
            continue

        # deletion
        if i > 0 and d[i][j] > d[i - 1][j]:
            operation.append(2)
            i -= 1
            continue

        # substition
        if d[i][j] > d[i - 1][j - 1]:
            operation.append(3)
        
        # nothing to do
        if d[i][j] == d[i - 1][j - 1]:
            operation.append(0)
        
        i -= 1
        j -= 1

    operation.reverse()
    return operation

word1 = 'deal'
word2 = 'leda'
dist, op_name, (word1, word2) = mini_edit_distance(word1, word2, [1,1,1])
print('minimum edit distance =', dist)
print('operation names =', op_name)
