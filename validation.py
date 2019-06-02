import torch
import time

def validate(classified, answer):

    start_time = time.time()
    
    count = 0
    correct = 0
    assert classified.shape[0] == len(answer)
    for index in range(classified.shape[0]):
        count += 1
        output = torch.argmax(classified[index])
        if output == answer[index]:
            correct += 1
    accuracy = correct * 100 / count
    
    end_time = time.time()
    time_elapsed = end_time - start_time
    print('Time Elapsed for This Validation: %02d:%02d:%02d' % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))
    
    return accuracy