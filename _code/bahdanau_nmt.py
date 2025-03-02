import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Config:
    def __init__(self):
        # 모델 하이퍼파라미터
        self.hidden_dim = 256
        self.embedding_dim = 256
        self.num_layers = 1
        
        # 학습 하이퍼파라미터
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 100
        self.max_seq_length = 50
        
        # 특수 토큰
        self.PAD_TOKEN = 0
        self.START_TOKEN = 1
        self.END_TOKEN = 2

class GRUCell:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        scale = 0.001
        self.Wz = np.random.randn(hidden_dim, input_dim) * scale
        self.Uz = np.random.randn(hidden_dim, hidden_dim) * scale
        self.bz = np.zeros((hidden_dim, 1))
        
        self.Wr = np.random.randn(hidden_dim, input_dim) * scale
        self.Ur = np.random.randn(hidden_dim, hidden_dim) * scale
        self.br = np.zeros((hidden_dim, 1))
        
        self.Wh = np.random.randn(hidden_dim, input_dim) * scale
        self.Uh = np.random.randn(hidden_dim, hidden_dim) * scale
        self.bh = np.zeros((hidden_dim, 1))

    def get_params(self):
        return {
            'Wz': self.Wz,
            'Uz': self.Uz,
            'bz': self.bz,
            'Wr': self.Wr,
            'Ur': self.Ur,
            'br': self.br,
            'Wh': self.Wh,
            'Uh': self.Uh,
            'bh': self.bh
        }
    
    def forward(self, x, h_prev):
        # 입력 reshape
        x = x.reshape(-1, 1)
        h_prev = h_prev.reshape(-1, 1)
        
        # 게이트 계산
        z = sigmoid(np.dot(self.Wz, x) + np.dot(self.Uz, h_prev) + self.bz)
        r = sigmoid(np.dot(self.Wr, x) + np.dot(self.Ur, h_prev) + self.br)
        
        # 후보 은닉 상태
        h_tilde = np.tanh(np.dot(self.Wh, x) + np.dot(self.Uh, (r * h_prev)) + self.bh)
        
        # 최종 은닉 상태
        h = (1 - z) * h_prev + z * h_tilde
        
        return h.reshape(-1)

class BidirectionalEncoder:
    def __init__(self, input_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        self.forward_gru = GRUCell(input_dim, hidden_dim)
        self.backward_gru = GRUCell(input_dim, hidden_dim)
    
    def forward(self, x, mask=None):
        seq_len = len(x)
        
        # 순방향 처리
        forward_states = []
        h_forward = np.zeros(self.hidden_dim)
        for t in range(seq_len):
            h_forward = self.forward_gru.forward(x[t], h_forward)
            forward_states.append(h_forward)
        
        # 역방향 처리
        backward_states = []
        h_backward = np.zeros(self.hidden_dim)
        for t in range(seq_len-1, -1, -1):
            h_backward = self.backward_gru.forward(x[t], h_backward)
            backward_states.insert(0, h_backward)
        
        # 양방향 상태 결합
        encoder_states = [np.concatenate([f, b]) 
                         for f, b in zip(forward_states, backward_states)]
        
        return np.array(encoder_states)

class BahdanauAttention:
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        scale = 0.01
        
        # 가중치 초기화
        self.W_a = np.random.randn(hidden_dim, hidden_dim) * scale
        self.U_a = np.random.randn(hidden_dim, 2*hidden_dim) * scale
        self.v_a = np.random.randn(hidden_dim) * scale
    
    def compute_attention(self, encoder_states, decoder_state):
        # 어텐션 스코어 계산
        scores = []
        for h_j in encoder_states:
            score = np.tanh(
                np.dot(self.W_a, decoder_state) + 
                np.dot(self.U_a, h_j)
            )
            score = np.dot(self.v_a, score)
            scores.append(score)
        
        # 소프트맥스로 확률 변환
        scores = np.array(scores)
        attention_weights = softmax(scores)
        
        # context 벡터 계산
        context = np.sum(encoder_states * attention_weights[:, np.newaxis], axis=0)
        
        return context, attention_weights

class AttentionDecoder:
    def __init__(self, hidden_dim, output_dim):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # GRU 입력 차원 수정: context vector(2*hidden_dim) + embedding_dim
        self.gru = GRUCell(3*hidden_dim, hidden_dim)  # 2*hidden_dim(context) + hidden_dim(embedding)
        
        # 출력층 가중치
        scale = 0.01
        self.W_out = np.random.randn(output_dim, hidden_dim) * scale
        self.b_out = np.zeros(output_dim)
    
    def forward(self, context, y_prev, s_prev):
        # GRU 입력 준비 (context: 2*hidden_dim, y_prev: hidden_dim)
        gru_input = np.concatenate([y_prev, context])
        
        # 다음 상태 계산
        s_next = self.gru.forward(gru_input, s_prev)
        
        # 출력 계산
        output = np.dot(self.W_out, s_next) + self.b_out
        
        return s_next, output

class BahdanauNMT:
    def __init__(self, config, source_vocab_size, target_vocab_size):
        self.config = config
        self.encoder = BidirectionalEncoder(config.embedding_dim, config.hidden_dim)
        self.attention = BahdanauAttention(config.hidden_dim)
        self.decoder = AttentionDecoder(config.hidden_dim, target_vocab_size)
        
        self.source_embedding = np.random.randn(source_vocab_size, config.embedding_dim) * 0.01
        self.target_embedding = np.random.randn(target_vocab_size, config.embedding_dim) * 0.01
        
        # 모델 파라미터를 딕셔너리로 저장
        self.params = {
            'source_embedding': self.source_embedding,
            'target_embedding': self.target_embedding,
            'encoder_forward_gru': self.encoder.forward_gru.get_params(),
            'encoder_backward_gru': self.encoder.backward_gru.get_params(),
            'decoder_gru': self.decoder.gru.get_params(),
            'W_out': self.decoder.W_out,
            'b_out': self.decoder.b_out
        }
    
    def forward(self, x, y=None, teacher_forcing=True):
        # 임베딩
        x_embedded = self.source_embedding[x]
        
        # 인코딩
        encoder_states = self.encoder.forward(x_embedded)
        
        # 디코딩 초기화
        decoder_state = np.zeros(self.config.hidden_dim)
        outputs = []
        attentions = []
        
        # 디코딩
        max_len = len(y) if y is not None else self.config.max_seq_length
        current_token = self.config.START_TOKEN
        
        for t in range(max_len):
            # 현재 토큰 임베딩
            if t == 0:
                decoder_input = np.zeros(self.config.hidden_dim)
            else:
                decoder_input = np.zeros(self.config.hidden_dim)
                if current_token < len(self.target_embedding):
                    decoder_input = self.target_embedding[current_token]
            
            # 어텐션 및 디코딩
            context, attention = self.attention.compute_attention(
                encoder_states, decoder_state)
            decoder_state, output = self.decoder.forward(
                context, decoder_input, decoder_state)
            
            outputs.append(output)
            attentions.append(attention)
            
            if teacher_forcing and y is not None:
                current_token = y[t]
            else:
                current_token = np.argmax(output)
                
            if current_token == self.config.END_TOKEN:
                break
        
        return np.array(outputs), np.array(attentions)

# 유틸리티 함수들
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def cross_entropy_loss(predictions, targets):
    batch_size = len(targets)
    probs = softmax(predictions)
    correct_logprobs = -np.log(probs[range(batch_size), targets])
    return np.mean(correct_logprobs)

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params):
        self.t += 1
        for key in params.keys():
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * params[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (params[key] ** 2)
            
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

def train(model, source_data, target_data, config):
    optimizer = AdamOptimizer(learning_rate=config.learning_rate)
    
    for epoch in range(config.num_epochs):
        total_loss = 0
        for i in range(len(source_data)):
            source_indices = [source_vocab[word] for word in source_data[i].split()]
            target_indices = [target_vocab[word] for word in target_data[i].split()]
            
            # 모델의 forward pass
            outputs, attentions = model.forward(source_indices, target_indices)
            
            # 손실 계산
            loss = cross_entropy_loss(outputs, target_indices)
            total_loss += loss
            
            # 가중치 업데이트
            # 모델의 파라미터를 평탄화하여 옵티마이저에 전달
            params = {
                'source_embedding': model.source_embedding,
                'target_embedding': model.target_embedding,
                'encoder_forward_gru': model.encoder.forward_gru.get_params(),
                'encoder_backward_gru': model.encoder.backward_gru.get_params(),
                'decoder_gru': model.decoder.gru.get_params(),
                'W_out': model.decoder.W_out,
                'b_out': model.decoder.b_out
            }
            
            # 각 파라미터를 NumPy 배열로 변환하여 옵티마이저에 전달
            flat_params = {key: value for key, value in params.items() if isinstance(value, np.ndarray)}
            optimizer.update(flat_params)  # Adam 옵티마이저를 사용하여 가중치 업데이트
            
        print(f"Epoch {epoch + 1}/{config.num_epochs}, Loss: {total_loss / len(source_data)}")

if __name__ == "__main__":
    # matplotlib 폰트 설정
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 샘플 데이터셋 생성
    source_sentences = [
        "나는 학교에 간다",
        "그녀는 책을 읽는다",
        "그는 음악을 듣는다",
        "우리는 공부를 한다"
    ]
    
    target_sentences = [
        "I go to school",
        "She reads a book",
        "He listens to music",
        "We study"
    ]
    
    # 간단한 토크나이저 구현
    def build_vocab(sentences):
        vocab = {'<PAD>': 0, '<START>': 1, '<END>': 2}
        for sentence in sentences:
            for word in sentence.split():
                if word not in vocab:
                    vocab[word] = len(vocab)
        return vocab
    
    # 데이터 전처리
    source_vocab = build_vocab(source_sentences)
    target_vocab = build_vocab(target_sentences)
    
    # 설정 초기화
    config = Config()
    
    # 모델 초기화
    model = BahdanauNMT(
        config=config,
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab)
    )
    
    # 어텐션 시각화를 위한 예시 문장
    def visualize_sample():
        source = "나는 학교에 간다"
        target = "I go to school"
        
        # 문장을 인덱스로 변환
        source_indices = [source_vocab[word] for word in source.split()]
        target_indices = [target_vocab[word] for word in target.split()]
        
        # 모델 실행
        outputs, attentions = model.forward(source_indices, target_indices)
        
        # 어텐션 시각화
        plt.figure(figsize=(10, 8))
        sns.heatmap(attentions,
                    xticklabels=source.split(),
                    yticklabels=target.split(),
                    cmap='viridis')
        plt.xlabel('Source Words')
        plt.ylabel('Target Words')
        plt.title('Attention Weights Visualization')
        plt.show()
    
    # 번역 예시
    def test_translations():
        test_sentences = [
            "나는 공부를 한다",
            "그녀는 음악을 듣는다",
            "그는 책을 읽는다"
        ]
        
        print("\n=== 번역 테스트 ===")
        for source in test_sentences:
            source_indices = [source_vocab[word] for word in source.split()]
            outputs, _ = model.forward(source_indices)
            
            # 출력을 단어로 변환
            rev_target_vocab = {v: k for k, v in target_vocab.items()}
            translation = []
            
            # 최대 10개 토큰까지만 생성
            for i, output in enumerate(outputs[:5]):
                token = np.argmax(output)
                if token == config.END_TOKEN:
                    break
                if token in rev_target_vocab:
                    translation.append(rev_target_vocab[token])
            
            print(f"입력: {source}")
            print(f"출력: {' '.join(translation)}")
            print("-" * 50)
    
    # 모델 훈련
    train(model, source_sentences, target_sentences, config)

    # 실행
    print("어텐션 시각화 예시:")
    visualize_sample()
    
    print("\n번역 테스트 실행:")
    test_translations()


