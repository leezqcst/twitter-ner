""" Preprocessing utils """
def chargram(token, n=3):
    """ Convert word into character level ngrams.
    
    We pad both ends of the word with _ tokens on both ends for `wide` ngrams
    
    Eg:
        
        ['__i', '_in', 'inp', 'npu', 'put', 'ut_', 't__'] = str_to_char_ngrams('input', 3)
    """
    token = '_'*(n-1) + token + '_'*(n-1)
    chargram = []
    for i in range(len(token[:-(n-1)])):
        chargram.append(token[i:i+n])
    return chargram

def chargrams(tokens, n=3):
    return [ chargram(token, n=n) for token in tokens ]

def pad_tensor(tensor, pad_symbol):
    if not isinstance(tensor[0], (list,tuple)):
        return tensor

    
    pad_len = max(len(sub_tensor) for sub_tensor in tensor)
    tensor = type(tensor)([ type(sub_tensor)(pad_tensor(
                                [element for element in sub_tensor]
                                +[[pad_symbol]*pad_len]*(pad_len-len(sub_tensor)), pad_symbol))
                            for sub_tensor in tensor])
    return tensor

def sentences_to_chargrams(sentences, vocab):
    all_xs, all_ws = [], []
    for sentence in sentences:
        xs, ws = [], []
        for token in sentence:
            grams = chargram(token)
            x = [ vocab.idx(cgram) for cgram in grams ]
            w = [ 1 for cgram in grams ]
            xs.append(x)
            ws.append(w)
        all_xs.append(xs)
        all_ws.append(ws)
    return all_xs, all_ws

def pad_chargrams(sentences, ws, pad):
    max_sent_len = max(len(sent) for sent in sentences)
    max_word_len = max(len(word) for sent in sentences for word in sent)
    padded_sentences, padded_ws = [], []
    for sentence, weights in zip(sentences, ws):
        padded_sentence = []
        padded_weights = []
        for token, weight in zip(sentence, weights):
            token += [pad]*(max_word_len-len(token))
            weight += [0]*(max_word_len-len(weight))
            padded_sentence.append(token)
            padded_weights.append(weight)
        for _ in range(max_sent_len - len(sentence)):
            padded_sentence.append([pad]*max_word_len)
            padded_weights.append([0]*max_word_len)
        padded_sentences.append(padded_sentence)
        padded_ws.append(padded_weights)
    return padded_sentences, padded_ws, max_sent_len, max_word_len

def sequence_to_index(sentence, vocab):
    return [ vocab.idx(token) for token in sentence ]

def sequences_to_indices(sentences, vocab):
    return [ sequence_to_index(sentence, vocab) for sentence in sentences ]

def index_to_sequence(sentence, vocab):
    return [ vocab.token(token) for token in sentence ]

def indices_to_sequences(sentences, vocab):
    return [ index_to_sequenceactivity(sentence, vocab) for sentence in sentences ]

def pad_sequence(sentence, pad, pad_len):
    padded_sequence = sentence + [pad]*(pad_len-len(sentence))
    pad_weight = [1]*len(sentence) + [0]*(pad_len - len(sentence))
    return padded_sequence, pad_weight

def pad_sequences(sentences, pad):
    pad_len = max(len(sentence) for sentence in sentences)
    pad_sequences, pad_weights = [], []
    for sequence in sentences:
        padded_sequence, pad_weight = pad_sequence(sequence, pad, pad_len)
        pad_sequences.append(padded_sequence)
        pad_weights.append(pad_weight)
    return pad_sequences, pad_weights, pad_len

def depad_sequence(sequence, pad):
    if isinstance(pad, list):
        return [ s for s, p in zip(sequence, pad) if p != 0 ]
    else:
        return [ s for s in sequence if s !=pad ]

def depad_sequences(sequences, pad):
    if isinstance(pad, list):
        return [ depad_sequence(sequence, p) for sequence, p in zip(sequences, pad) ]
    else:
        return [ depad_sequence(sequence, pad) for sequence in sequences ]