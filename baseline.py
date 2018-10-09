def one_step_baseline_prediction(train, test):
    history = [x for x in train]
    predictions = list()

    for i in range(len(test)):
        predictions.append(history[-1])
        history.append(test[i])

    return predictions

def multistep_baseline_prediction(train, test, lag):
    results = []
    for i in range(-5, 0, 1):
        results.append([train[i] for j in range(lag)])
    
    for i in range(len(test) - lag):
        results.append([test[i] for j in range(lag)])
    return results