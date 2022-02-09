import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#######
### FRAMM Implementations
#######

class DiagnosisEncoder(nn.Module):
  def __init__(self, claim_dim, lstm_dim, embedding_dim, num_layers):
    super(DiagnosisEncoder, self).__init__()
    self.claim_dim = claim_dim
    self.lstm_dim = lstm_dim
    self.embedding_dim = embedding_dim
    self.num_layers = num_layers
    self.biLSTM = nn.LSTM(claim_dim, lstm_dim, num_layers, batch_first=True, bidirectional=True)
    self.fc = nn.Linear(2*lstm_dim, embedding_dim)
    self.firstInput = nn.Parameter(torch.rand(1,1,1,claim_dim, dtype=torch.float))

  def forward(self, input, lengths, num_inv):
    bs = input.size(0)
    input = torch.cat((self.firstInput.repeat(bs, num_inv, 1, 1), input), -2)
    input = torch.reshape(input, (bs * num_inv, -1, self.claim_dim))
    lengths = torch.reshape(lengths, (bs * num_inv,))
    packed_input = pack_padded_sequence(input, lengths+1, batch_first=True, enforce_sorted=False)
    packed_output, _ = self.biLSTM(packed_input)
    out, _ = pad_packed_sequence(packed_output, batch_first=True)
    combined_out = torch.cat((out[:, 0, self.lstm_dim:], out[range(len(out)), lengths, :self.lstm_dim]), 1)
    combined_out = torch.reshape(combined_out, (bs, num_inv, 2 * self.lstm_dim))
    return(self.fc(combined_out))

class PrescriptionEncoder(nn.Module):
  def __init__(self, claim_dim, lstm_dim, embedding_dim, num_layers):
    super(PrescriptionEncoder, self).__init__()
    self.claim_dim = claim_dim
    self.lstm_dim = lstm_dim
    self.embedding_dim = embedding_dim
    self.num_layers = num_layers
    self.biLSTM = nn.LSTM(claim_dim, lstm_dim, num_layers, batch_first=True, bidirectional=True)
    self.fc = nn.Linear(2*lstm_dim, embedding_dim)
    self.firstInput = nn.Parameter(torch.rand(1,1,1,claim_dim, dtype=torch.float))

  def forward(self, input, lengths, num_inv):
    bs = input.size(0)
    input = torch.cat((self.firstInput.repeat(bs, num_inv, 1, 1), input), -2)
    input = torch.reshape(input, (bs * num_inv, -1, self.claim_dim))
    lengths = torch.reshape(lengths, (bs * num_inv,))
    packed_input = pack_padded_sequence(input, lengths+1, batch_first=True, enforce_sorted=False)
    packed_output, _ = self.biLSTM(packed_input)
    out, _ = pad_packed_sequence(packed_output, batch_first=True)
    combined_out = torch.cat((out[:, 0, self.lstm_dim:], out[range(len(out)), lengths, :self.lstm_dim]), 1)
    combined_out = torch.reshape(combined_out, (bs, num_inv, 2 * self.lstm_dim))
    return(self.fc(combined_out))

class PastTrialEncoder(nn.Module):
  def __init__(self, trial_dim, lstm_dim, embedding_dim, num_layers):
    super(PastTrialEncoder, self).__init__()
    self.trial_dim = trial_dim
    self.lstm_dim = lstm_dim
    self.embedding_dim = embedding_dim
    self.num_layers = num_layers
    self.biLSTM = nn.LSTM(trial_dim+1-(2*768), lstm_dim, num_layers, batch_first=True, bidirectional=True)
    self.fc = nn.Linear(2*lstm_dim, embedding_dim)
    self.firstInput = nn.Parameter(torch.rand(1,1,1,trial_dim+1-(2*768), dtype=torch.float))

  def forward(self, input, lengths, num_inv):
    bs = input.size(0)
    input = torch.cat((self.firstInput.repeat(bs, num_inv, 1, 1), input), -2)
    input = torch.reshape(input, (bs * num_inv, -1, self.trial_dim+1-(2*768)))
    lengths = torch.reshape(lengths, (bs * num_inv,))
    packed_input = pack_padded_sequence(input, lengths+1, batch_first=True, enforce_sorted=False)
    packed_output, _ = self.biLSTM(packed_input)
    out, _ = pad_packed_sequence(packed_output, batch_first=True)
    combined_out = torch.cat((out[:, 0, self.lstm_dim:], out[range(len(out)), lengths, :self.lstm_dim]), -1)
    combined_out = torch.reshape(combined_out, (bs, num_inv, 2 * self.lstm_dim))
    return(self.fc(combined_out))
  
class AttentionEncoder(nn.Module):
  def __init__(self, embed_dim, n_keys, n_heads):
    super(AttentionEncoder, self).__init__()
    self.embed_dim = embed_dim
    self.attention = nn.MultiheadAttention(embed_dim, n_heads)#, batch_first=True)

  def forward(self, query, values, attention_mask):
    bs = values.size(0)
    num_inv = values.size(1)
    query = torch.reshape(query, (bs * num_inv, 1, self.embed_dim)).transpose(0,1)
    values = torch.reshape(values, (bs * num_inv, -1, self.embed_dim)).transpose(0,1)
    attention_mask = torch.reshape(attention_mask, (bs * num_inv, -1)).bool()
    embeddings, _ = self.attention(query, values, values, attention_mask, need_weights=False)
    embeddings = torch.reshape(embeddings.transpose(0,1), (bs, num_inv, self.embed_dim))
    return embeddings
  
  
  
class MCAT(nn.Module):
  def __init__(self, trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers, hidden_dim, n_heads):
    super(MCAT, self).__init__()
    self.static_encoder = nn.Linear(static_dim, embedding_dim)
    self.diagnosis_encoder = DiagnosisEncoder(dx_dim, lstm_dim, embedding_dim, num_layers)
    self.prescription_encoder = PrescriptionEncoder(rx_dim, lstm_dim, embedding_dim, num_layers)
    self.history_encoder = PastTrialEncoder(trial_dim, lstm_dim, embedding_dim, num_layers)
    self.trial_encoder = nn.Linear(trial_dim, embedding_dim)
    self.stat_fc = nn.Linear(embedding_dim, embedding_dim)
    self.dx_fc = nn.Linear(embedding_dim, embedding_dim)
    self.rx_fc = nn.Linear(embedding_dim, embedding_dim)
    self.hist_fc = nn.Linear(embedding_dim, embedding_dim)
    self.trial_fc = nn.Linear(embedding_dim, embedding_dim)
    self.attention_encoder = AttentionEncoder(embedding_dim, 4, n_heads)
    encoder_layer = nn.TransformerEncoderLayer(2*embedding_dim, n_heads, hidden_dim)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 2)
    self.fc = nn.Linear(2*embedding_dim, hidden_dim)
    self.output = nn.Linear(hidden_dim, 1)

  def forward(self, trial, investigators, inv_dx, inv_dx_lens, inv_rx, inv_rx_lens, past_trials, past_trials_lengths, inv_mask, num_inv):
    # Trial is (bs, trial_dim)
    # All other inputs are (bs, M, *) where * is either a sequence or single input
    investigator_encoding = self.stat_fc(torch.relu(self.static_encoder(investigators)))
    dx_encoding = self.dx_fc(torch.relu(self.diagnosis_encoder(inv_dx, inv_dx_lens, num_inv)))
    rx_encoding = self.rx_fc(torch.relu(self.prescription_encoder(inv_rx, inv_rx_lens, num_inv)))
    history_encoding = self.hist_fc(torch.relu(self.history_encoder(past_trials, past_trials_lengths, num_inv)))
    trial_encoding = self.trial_fc(torch.relu(self.trial_encoder(trial)))
    trial_encoding = trial_encoding.unsqueeze(1).repeat(1, num_inv, 1)
    inv_representation = self.attention_encoder(trial_encoding, torch.stack([investigator_encoding, dx_encoding, rx_encoding, history_encoding], dim=2), inv_mask)
    inv_site_representation = torch.cat((inv_representation, trial_encoding), dim=-1)
    network_input = self.transformer_encoder(inv_site_representation.transpose(0,1)).transpose(0,1)
    score = self.output(torch.relu(self.fc(network_input)))
    return score



#######
### Baseline Methods
#######

class MCAT_FC(nn.Module):
  def __init__(self, trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers, hidden_dim, n_heads):
    super(MCAT_FC, self).__init__()
    self.static_encoder = nn.Linear(static_dim, embedding_dim)
    self.diagnosis_encoder = DiagnosisEncoder(dx_dim, lstm_dim, embedding_dim, num_layers)
    self.prescription_encoder = PrescriptionEncoder(rx_dim, lstm_dim, embedding_dim, num_layers)
    self.history_encoder = PastTrialEncoder(trial_dim, lstm_dim, embedding_dim, num_layers)
    self.trial_encoder = nn.Linear(trial_dim, embedding_dim)
    self.attention_encoder = AttentionEncoder(embedding_dim, 4, n_heads)
    
    self.fc = nn.Linear(2*embedding_dim, embedding_dim)
    self.fc2 = nn.Linear(embedding_dim, hidden_dim)
    self.output = nn.Linear(hidden_dim, 1)

  def forward(self, trial, investigators, inv_dx, inv_dx_lens, inv_rx, inv_rx_lens, past_trials, past_trials_lengths, inv_mask, num_inv):
    # Trial is (bs, trial_dim)
    # All other inputs are (bs, M, *) where * is either a sequence or single input
    investigator_encoding = self.static_encoder(investigators)
    dx_encoding = self.diagnosis_encoder(inv_dx, inv_dx_lens, num_inv)
    rx_encoding = self.prescription_encoder(inv_rx, inv_rx_lens, num_inv)
    history_encoding = self.history_encoder(past_trials, past_trials_lengths, num_inv)
    trial_encoding = self.trial_encoder(trial)
    trial_encoding = trial_encoding.unsqueeze(1).repeat(1, num_inv, 1)
    inv_representation = self.attention_encoder(trial_encoding, torch.stack([investigator_encoding, dx_encoding, rx_encoding, history_encoding], dim=2), inv_mask)
    inv_site_representation = torch.cat((inv_representation, trial_encoding), dim=-1)
    score = self.output(torch.relu(self.fc2(torch.relu(self.fc(inv_site_representation)))))
    return score



class ModalityDropout(nn.Module):
  def __init__(self, trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers, hidden_dim, n_heads):
    super(ModalityDropout, self).__init__()
    self.static_encoder = nn.Linear(static_dim, embedding_dim)
    self.diagnosis_encoder = DiagnosisEncoder(dx_dim, lstm_dim, embedding_dim, num_layers)
    self.prescription_encoder = PrescriptionEncoder(rx_dim, lstm_dim, embedding_dim, num_layers)
    self.history_encoder = PastTrialEncoder(trial_dim, lstm_dim, embedding_dim, num_layers)
    self.trial_encoder = nn.Linear(trial_dim, embedding_dim)
    self.inv_encoder = nn.Linear(5 * embedding_dim, 2 * embedding_dim)
    
    encoder_layer = nn.TransformerEncoderLayer(2*embedding_dim, n_heads, hidden_dim)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 2)
    self.fc = nn.Linear(2*embedding_dim, hidden_dim)
    self.output = nn.Linear(hidden_dim, 1)

  def forward(self, trial, investigators, inv_dx, inv_dx_lens, inv_rx, inv_rx_lens, past_trials, past_trials_lengths, inv_mask, num_inv):
    # Trial is (bs, trial_dim)
    # All other inputs are (bs, M, *) where * is either a sequence or single input
    investigator_encoding = self.static_encoder(investigators) * (1 - inv_mask[:,:,0].unsqueeze(-1))
    dx_encoding = self.diagnosis_encoder(inv_dx, inv_dx_lens, num_inv) * (1 - inv_mask[:,:,1].unsqueeze(-1))
    rx_encoding = self.prescription_encoder(inv_rx, inv_rx_lens, num_inv) * (1 - inv_mask[:,:,2].unsqueeze(-1))
    history_encoding = self.history_encoder(past_trials, past_trials_lengths, num_inv) * (1 - inv_mask[:,:,3].unsqueeze(-1))
    trial_encoding = self.trial_encoder(trial)
    trial_encoding = trial_encoding.unsqueeze(1).repeat(1, num_inv, 1)
    inv_site_representation = self.inv_encoder(torch.cat((trial_encoding, investigator_encoding, dx_encoding, rx_encoding, history_encoding), dim=-1))
    network_input = self.transformer_encoder(inv_site_representation.transpose(0,1)).transpose(0,1)
    score = self.output(torch.relu(self.fc(network_input)))
    return score



class DiagnosisDecoder(nn.Module):
  def __init__(self, seq_len, claim_dim, lstm_dim, embedding_dim, num_layers):
    super(DiagnosisDecoder, self).__init__()
    self.seq_len = seq_len
    self.claim_dim = claim_dim
    self.lstm_dim = lstm_dim
    self.embedding_dim = embedding_dim
    self.num_layers = num_layers
    self.lstm = nn.LSTM(embedding_dim, lstm_dim, num_layers, batch_first=True)
    self.fc = nn.Linear(lstm_dim, claim_dim)

  def forward(self, inv_representation):
    bs = inv_representation.size(0)
    input = inv_representation.unsqueeze(2).repeat(1, 1, self.seq_len, 1)
    out, _ = self.lstm(input.view(-1, self.seq_len, self.embedding_dim))
    out = out.reshape(bs, -1, self.seq_len, self.lstm_dim)
    out = self.fc(out)
    soft = nn.Softmax(-1)
    out = soft(out)
    return out

class PrescriptionDecoder(nn.Module):
  def __init__(self, seq_len, claim_dim, lstm_dim, embedding_dim, num_layers):
    super(PrescriptionDecoder, self).__init__()
    self.seq_len = seq_len
    self.claim_dim = claim_dim
    self.lstm_dim = lstm_dim
    self.embedding_dim = embedding_dim
    self.num_layers = num_layers
    self.lstm = nn.LSTM(embedding_dim, lstm_dim, num_layers, batch_first=True)
    self.fc = nn.Linear(lstm_dim, claim_dim)

  def forward(self, inv_representation):
    bs = inv_representation.size(0)
    input = inv_representation.unsqueeze(2).repeat(1, 1, self.seq_len, 1)
    out, _ = self.lstm(input.view(-1, self.seq_len, self.embedding_dim))
    out = out.reshape(bs, -1, self.seq_len, self.lstm_dim)
    out = self.fc(out)
    soft = nn.Softmax(-1)
    out = soft(out)
    return out

class PastTrialDecoder(nn.Module):
  def __init__(self, seq_len, trial_dim, lstm_dim, embedding_dim, num_layers):
    super(PastTrialDecoder, self).__init__()
    self.seq_len = seq_len
    self.trial_dim = trial_dim
    self.lstm_dim = lstm_dim
    self.embedding_dim = embedding_dim
    self.num_layers = num_layers
    self.lstm = nn.LSTM(embedding_dim, lstm_dim, num_layers, batch_first=True)
    self.fc = nn.Linear(lstm_dim, trial_dim+1-(2*768))

  def forward(self, inv_representation):
    bs = inv_representation.size(0)
    input = inv_representation.unsqueeze(2).repeat(1, 1, self.seq_len, 1)
    out, _ = self.lstm(input.view(-1, self.seq_len, self.embedding_dim))
    out = out.reshape(bs, -1, self.seq_len, self.lstm_dim)
    out = self.fc(out)
    return out

class UnifiedRepresentationEncoder(nn.Module):
  def __init__(self, trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers):
    super(UnifiedRepresentationEncoder, self).__init__()
    self.embedding_dim = embedding_dim
    self.static_encoder = nn.Linear(static_dim, embedding_dim)
    self.diagnosis_encoder = DiagnosisEncoder(dx_dim, lstm_dim, embedding_dim, num_layers)
    self.prescription_encoder = PrescriptionEncoder(rx_dim, lstm_dim, embedding_dim, num_layers)
    self.history_encoder = PastTrialEncoder(trial_dim, lstm_dim, embedding_dim, num_layers)
    self.static_bn = nn.BatchNorm1d(embedding_dim, affine=False)
    self.diagnosis_bn = nn.BatchNorm1d(embedding_dim, affine=False)
    self.prescription_bn = nn.BatchNorm1d(embedding_dim, affine=False)
    self.history_bn = nn.BatchNorm1d(embedding_dim, affine=False)

  def forward(self, investigators, inv_dx, inv_dx_lens, inv_rx, inv_rx_lens, past_trials, past_trials_lengths, inv_mask, num_inv):
    # Trial is (bs, trial_dim)
    # All other inputs are (bs, M, *) where * is either a sequence or single input
    investigator_encoding = self.static_encoder(investigators)
    dx_encoding = self.diagnosis_encoder(inv_dx, inv_dx_lens, num_inv)
    rx_encoding = self.prescription_encoder(inv_rx, inv_rx_lens, num_inv)
    history_encoding = self.history_encoder(past_trials, past_trials_lengths, num_inv)
    investigator_encoding = self.static_bn(investigator_encoding.view(-1, self.embedding_dim)).view(-1, num_inv, self.embedding_dim) * (1 - inv_mask[:,:,0].unsqueeze(-1))
    dx_encoding = self.diagnosis_bn(dx_encoding.view(-1, self.embedding_dim)).view(-1, num_inv, self.embedding_dim) * (1 - inv_mask[:,:,1].unsqueeze(-1))
    rx_encoding = self.prescription_bn(rx_encoding.view(-1, self.embedding_dim)).view(-1, num_inv, self.embedding_dim) * (1 - inv_mask[:,:,2].unsqueeze(-1))
    history_encoding = self.history_bn(history_encoding.view(-1, self.embedding_dim)).view(-1, num_inv, self.embedding_dim) * (1 - inv_mask[:,:,3].unsqueeze(-1))
    inv_representation = (investigator_encoding + dx_encoding + rx_encoding + history_encoding) / (4 - torch.sum(inv_mask, dim = -1, keepdim = True))
    
    return inv_representation

class UnifiedRepresentationDecoder(nn.Module):
  def __init__(self, claim_seq_len, trial_seq_len, trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers):
    super(UnifiedRepresentationDecoder, self).__init__()
    self.dx_dim = dx_dim
    self.rx_dim = rx_dim
    self.claim_seq_len = claim_seq_len
    self.trial_seq_len = trial_seq_len
    self.static_decoder = nn.Linear(embedding_dim, static_dim)
    self.diagnosis_decoder = DiagnosisDecoder(claim_seq_len, dx_dim, lstm_dim, embedding_dim, num_layers)
    self.prescription_decoder = PrescriptionDecoder(claim_seq_len, rx_dim, lstm_dim, embedding_dim, num_layers)
    self.history_decoder = PastTrialDecoder(trial_seq_len, trial_dim, lstm_dim, embedding_dim, num_layers)

  def forward(self, inv_representation, investigators, inv_dx, inv_dx_lens, inv_rx, inv_rx_lens, past_trials, past_trials_lengths, inv_mask, num_inv):
    # Trial is (bs, trial_dim)
    # All other inputs are (bs, M, *) where * is either a sequence or single input
    static_decoding = self.static_decoder(inv_representation)
    dx_decoding = self.diagnosis_decoder(inv_representation)
    rx_decoding = self.prescription_decoder(inv_representation)
    history_decoding = self.history_decoder(inv_representation)

    mse = nn.MSELoss(reduction='none')
    ce = nn.CrossEntropyLoss(reduction='none')

    dx_label = inv_dx.nonzero(as_tuple=True)[-1]
    rx_label = inv_rx.nonzero(as_tuple=True)[-1]

    inv_dx_lens_binary = inv_dx_lens.bool().int().unsqueeze(-1).to(device)
    inv_rx_lens_binary = inv_rx_lens.bool().int().unsqueeze(-1).to(device)
    past_trial_lengths_binary = past_trials_lengths.bool().int().unsqueeze(-1).to(device)
    
    inv_dx_lens_safe = deepcopy(inv_dx_lens).cpu().type(torch.int64)
    inv_rx_lens_safe = deepcopy(inv_rx_lens).cpu().type(torch.int64)
    past_trial_lengths_safe = deepcopy(past_trials_lengths).cpu().type(torch.int64)
    inv_dx_lens_safe[inv_dx_lens_safe == 0] = 1
    inv_rx_lens_safe[inv_rx_lens_safe == 0] = 1
    past_trial_lengths_safe[past_trial_lengths_safe == 0] = 1

    static_loss = torch.mean(mse(static_decoding, investigators) * (1 - inv_mask[:,:,0].unsqueeze(-1)))
    dx_loss = torch.mean(pack_padded_sequence((ce(dx_decoding.view(-1, self.dx_dim), dx_label).view(-1, num_inv, self.claim_seq_len) * (1 - inv_mask[:,:,1].unsqueeze(-1)) * inv_dx_lens_binary).view(-1, self.claim_seq_len), inv_dx_lens_safe.flatten(), batch_first=True, enforce_sorted=False).data)
    rx_loss = torch.mean(pack_padded_sequence((ce(rx_decoding.view(-1, self.rx_dim), rx_label).view(-1, num_inv, self.claim_seq_len) * (1 - inv_mask[:,:,2].unsqueeze(-1)) * inv_rx_lens_binary).view(-1, self.claim_seq_len), inv_rx_lens_safe.flatten(), batch_first=True, enforce_sorted=False).data)
    history_loss = torch.mean(pack_padded_sequence((mse(history_decoding, past_trials).mean(-1) * (1 - inv_mask[:,:,3].unsqueeze(-1)) * past_trial_lengths_binary.squeeze(-1)).view(-1, self.trial_seq_len), past_trial_lengths_safe.flatten(), batch_first=True, enforce_sorted=False).data)

    loss = static_loss + dx_loss + rx_loss + history_loss
    return loss

class UnifiedRepresentationNetwork(nn.Module):
  def __init__(self, claim_seq_len, trial_seq_len, trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers, hidden_dim, n_heads):
    super(UnifiedRepresentationNetwork, self).__init__()
    self.encoder = UnifiedRepresentationEncoder(trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers)
    self.decoder = UnifiedRepresentationDecoder(claim_seq_len, trial_seq_len, trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers)
    self.trial_encoder = nn.Linear(trial_dim, embedding_dim)
    
    encoder_layer = nn.TransformerEncoderLayer(2*embedding_dim, n_heads, hidden_dim)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 2)
    self.fc = nn.Linear(2*embedding_dim, hidden_dim)
    self.output = nn.Linear(hidden_dim, 1)

  def autoencode(self, investigators, inv_dx, inv_dx_lens, inv_rx, inv_rx_lens, past_trials, past_trials_lengths, inv_mask, num_inv):
    inv_representation = self.encoder(investigators, inv_dx, inv_dx_lens, inv_rx, inv_rx_lens, past_trials, past_trials_lengths, inv_mask, num_inv)
    loss = self.decoder(inv_representation, investigators, inv_dx, inv_dx_lens, inv_rx, inv_rx_lens, past_trials, past_trials_lengths, inv_mask, num_inv)
    return loss

  def forward(self, trial, investigators, inv_dx, inv_dx_lens, inv_rx, inv_rx_lens, past_trials, past_trials_lengths, inv_mask, num_inv):
    # Trial is (bs, trial_dim)
    # All other inputs are (bs, M, *) where * is either a sequence or single input
    with torch.no_grad():
      self.encoder.eval()
      inv_representation = self.encoder(investigators, inv_dx, inv_dx_lens, inv_rx, inv_rx_lens, past_trials, past_trials_lengths, inv_mask, num_inv)

    trial_encoding = self.trial_encoder(trial)
    trial_encoding = trial_encoding.unsqueeze(1).repeat(1, num_inv, 1)
    inv_site_representation = torch.cat((inv_representation, trial_encoding), dim=-1)
    network_input = self.transformer_encoder(inv_site_representation.transpose(0,1)).transpose(0,1)
    score = self.output(torch.relu(self.fc(network_input)))
    return score



class ResidualAutoencoder(nn.Module):
  def __init__(self, claim_seq_len, trial_seq_len, trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers):
    super(ResidualAutoencoder, self).__init__()
    self.dx_dim = dx_dim
    self.rx_dim = rx_dim
    self.claim_seq_len = claim_seq_len
    self.trial_seq_len = trial_seq_len
    self.static_encoder = nn.Linear(static_dim, embedding_dim)
    self.diagnosis_encoder = DiagnosisEncoder(dx_dim, lstm_dim, embedding_dim, num_layers)
    self.prescription_encoder = PrescriptionEncoder(rx_dim, lstm_dim, embedding_dim, num_layers)
    self.history_encoder = PastTrialEncoder(trial_dim, lstm_dim, embedding_dim, num_layers)
    self.static_decoder = nn.Linear(4 * embedding_dim, static_dim)
    self.diagnosis_decoder = DiagnosisDecoder(claim_seq_len, dx_dim, lstm_dim, 4 * embedding_dim, num_layers)
    self.prescription_decoder = PrescriptionDecoder(claim_seq_len, rx_dim, lstm_dim, 4 * embedding_dim, num_layers)
    self.history_decoder = PastTrialDecoder(trial_seq_len, trial_dim, lstm_dim, 4 * embedding_dim, num_layers)

  def forward(self, investigators, inv_dx, inv_dx_lens, inv_rx, inv_rx_lens, past_trials, past_trials_lengths, num_inv):
    # Trial is (bs, trial_dim)
    # All other inputs are (bs, M, *) where * is either a sequence or single input
    investigator_encoding = self.static_encoder(investigators)
    dx_encoding = self.diagnosis_encoder(inv_dx, inv_dx_lens, num_inv)
    rx_encoding = self.prescription_encoder(inv_rx, inv_rx_lens, num_inv)
    history_encoding = self.history_encoder(past_trials, past_trials_lengths, num_inv)
    inv_representation = torch.cat((investigator_encoding, dx_encoding, rx_encoding, history_encoding), dim = -1)
    static_decoding = self.static_decoder(inv_representation)
    dx_decoding = self.diagnosis_decoder(inv_representation)
    rx_decoding = self.prescription_decoder(inv_representation)
    history_decoding = self.history_decoder(inv_representation)
    return static_decoding, dx_decoding, rx_decoding, history_decoding

class CascadedResidualAutoencoder(nn.Module):
  def __init__(self, claim_seq_len, trial_seq_len, trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers):
    super(CascadedResidualAutoencoder, self).__init__()
    self.ae1 = ResidualAutoencoder(claim_seq_len, trial_seq_len, trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers)
    self.ae2 = ResidualAutoencoder(claim_seq_len, trial_seq_len, trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers)
    self.ae3 = ResidualAutoencoder(claim_seq_len, trial_seq_len, trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers)

  def forward(self, investigators, inv_dx, inv_dx_lens, inv_rx, inv_rx_lens, past_trials, past_trials_lengths, inv_mask, num_inv):
    # Trial is (bs, trial_dim)
    # All other inputs are (bs, M, *) where * is either a sequence or single input
    input_inv = investigators * (1 - inv_mask[:,:,0].unsqueeze(-1))
    input_dx = inv_dx * (1 - inv_mask[:,:,1].unsqueeze(-1).unsqueeze(-1))
    input_rx = inv_rx * (1 - inv_mask[:,:,2].unsqueeze(-1).unsqueeze(-1))
    input_hist = past_trials * (1 - inv_mask[:,:,3].unsqueeze(-1).unsqueeze(-1))

    inv_1, dx_1, rx_1, hist_1 = self.ae1(input_inv, input_dx, inv_dx_lens, input_rx, inv_rx_lens, input_hist, past_trials_lengths, num_inv)
    inv_1 = inv_1 + input_inv
    dx_1 = dx_1 + input_dx
    rx_1 = rx_1 + input_rx
    hist_1 = hist_1 + input_hist

    inv_2, dx_2, rx_2, hist_2 = self.ae2(inv_1, dx_1, inv_dx_lens, rx_1, inv_rx_lens, hist_1, past_trials_lengths, num_inv)
    inv_2 = inv_2 + inv_1
    dx_2 = dx_2 + dx_1
    rx_2 = rx_2 + rx_1
    hist_2 = hist_2 + hist_1

    inv_3, dx_3, rx_3, hist_3 = self.ae3(inv_2, dx_2, inv_dx_lens, rx_2, inv_rx_lens, hist_2, past_trials_lengths, num_inv)
    inv_3 = inv_3 + inv_2
    dx_3 = dx_3 + dx_2
    rx_3 = rx_3 + rx_2
    hist_3 = hist_3 + hist_2
    return ((inv_1, dx_1, rx_1, hist_1), (inv_2, dx_2, rx_2, hist_2), (inv_3, dx_3, rx_3, hist_3)), (inv_3, dx_3, rx_3, hist_3)

class CascadedResidualAutoencoderNetwork(nn.Module):
  def __init__(self, claim_seq_len, trial_seq_len, trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers, hidden_dim, n_heads):
    super(CascadedResidualAutoencoderNetwork, self).__init__()
    self.imputer = CascadedResidualAutoencoder(claim_seq_len, trial_seq_len, trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers)
    
    self.dx_dim = dx_dim
    self.rx_dim = rx_dim
    self.claim_seq_len = claim_seq_len
    self.trial_seq_len = trial_seq_len

    self.static_encoder = nn.Linear(static_dim, embedding_dim)
    self.diagnosis_encoder = DiagnosisEncoder(dx_dim, lstm_dim, embedding_dim, num_layers)
    self.prescription_encoder = PrescriptionEncoder(rx_dim, lstm_dim, embedding_dim, num_layers)
    self.history_encoder = PastTrialEncoder(trial_dim, lstm_dim, embedding_dim, num_layers)
    self.trial_encoder = nn.Linear(trial_dim, embedding_dim)
    self.inv_encoder = nn.Linear(5 * embedding_dim, 2 * embedding_dim)
    
    encoder_layer = nn.TransformerEncoderLayer(2*embedding_dim, n_heads, hidden_dim)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 2)
    self.fc = nn.Linear(2*embedding_dim, hidden_dim)
    self.output = nn.Linear(hidden_dim, 1)

  def autoencode(self, investigators, inv_dx, inv_dx_lens, inv_rx, inv_rx_lens, past_trials, past_trials_lengths, inv_mask, num_inv):
    ((inv1, dx1, rx1, hist1), (inv2, dx2, rx2, hist2), (inv3, dx3, rx3, hist3)), _ = self.imputer(investigators, inv_dx, inv_dx_lens, inv_rx, inv_rx_lens, past_trials, past_trials_lengths, inv_mask, num_inv)
    
    mse = nn.MSELoss(reduction='none')
    ce = nn.CrossEntropyLoss(reduction='none')

    dx_label = inv_dx.nonzero(as_tuple=True)[-1]
    rx_label = inv_rx.nonzero(as_tuple=True)[-1]

    inv_dx_lens_binary = inv_dx_lens.bool().int().unsqueeze(-1).to(device)
    inv_rx_lens_binary = inv_rx_lens.bool().int().unsqueeze(-1).to(device)
    past_trial_lengths_binary = past_trials_lengths.bool().int().unsqueeze(-1).to(device)
    
    inv_dx_lens_safe = deepcopy(inv_dx_lens).cpu().type(torch.int64)
    inv_rx_lens_safe = deepcopy(inv_rx_lens).cpu().type(torch.int64)
    past_trial_lengths_safe = deepcopy(past_trials_lengths).cpu().type(torch.int64)
    inv_dx_lens_safe[inv_dx_lens_safe == 0] = 1
    inv_rx_lens_safe[inv_rx_lens_safe == 0] = 1
    past_trial_lengths_safe[past_trial_lengths_safe == 0] = 1

    static_loss1 = torch.mean(mse(inv1, investigators) * (1 - inv_mask[:,:,0].unsqueeze(-1)))
    dx_loss1 = torch.mean(pack_padded_sequence((ce(dx1.view(-1, self.dx_dim), dx_label).view(-1, num_inv, self.claim_seq_len) * (1 - inv_mask[:,:,1].unsqueeze(-1)) * inv_dx_lens_binary).view(-1, self.claim_seq_len), inv_dx_lens_safe.flatten(), batch_first=True, enforce_sorted=False).data)
    rx_loss1 = torch.mean(pack_padded_sequence((ce(rx1.view(-1, self.rx_dim), rx_label).view(-1, num_inv, self.claim_seq_len) * (1 - inv_mask[:,:,2].unsqueeze(-1)) * inv_rx_lens_binary).view(-1, self.claim_seq_len), inv_rx_lens_safe.flatten(), batch_first=True, enforce_sorted=False).data)
    history_loss1 = torch.mean(pack_padded_sequence((mse(hist1, past_trials).mean(-1) * (1 - inv_mask[:,:,3].unsqueeze(-1)) * past_trial_lengths_binary.squeeze(-1)).view(-1, self.trial_seq_len), past_trial_lengths_safe.flatten(), batch_first=True, enforce_sorted=False).data)

    static_loss2 = torch.mean(mse(inv2, investigators) * (1 - inv_mask[:,:,0].unsqueeze(-1)))
    dx_loss2 = torch.mean(pack_padded_sequence((ce(dx2.view(-1, self.dx_dim), dx_label).view(-1, num_inv, self.claim_seq_len) * (1 - inv_mask[:,:,1].unsqueeze(-1)) * inv_dx_lens_binary).view(-1, self.claim_seq_len), inv_dx_lens_safe.flatten(), batch_first=True, enforce_sorted=False).data)
    rx_loss2 = torch.mean(pack_padded_sequence((ce(rx2.view(-1, self.rx_dim), rx_label).view(-1, num_inv, self.claim_seq_len) * (1 - inv_mask[:,:,2].unsqueeze(-1)) * inv_rx_lens_binary).view(-1, self.claim_seq_len), inv_rx_lens_safe.flatten(), batch_first=True, enforce_sorted=False).data)
    history_loss2 = torch.mean(pack_padded_sequence((mse(hist2, past_trials).mean(-1) * (1 - inv_mask[:,:,3].unsqueeze(-1)) * past_trial_lengths_binary.squeeze(-1)).view(-1, self.trial_seq_len), past_trial_lengths_safe.flatten(), batch_first=True, enforce_sorted=False).data)

    static_loss3 = torch.mean(mse(inv3, investigators) * (1 - inv_mask[:,:,0].unsqueeze(-1)))
    dx_loss3 = torch.mean(pack_padded_sequence((ce(dx3.view(-1, self.dx_dim), dx_label).view(-1, num_inv, self.claim_seq_len) * (1 - inv_mask[:,:,1].unsqueeze(-1)) * inv_dx_lens_binary).view(-1, self.claim_seq_len), inv_dx_lens_safe.flatten(), batch_first=True, enforce_sorted=False).data)
    rx_loss3 = torch.mean(pack_padded_sequence((ce(rx3.view(-1, self.rx_dim), rx_label).view(-1, num_inv, self.claim_seq_len) * (1 - inv_mask[:,:,2].unsqueeze(-1)) * inv_rx_lens_binary).view(-1, self.claim_seq_len), inv_rx_lens_safe.flatten(), batch_first=True, enforce_sorted=False).data)
    history_loss3 = torch.mean(pack_padded_sequence((mse(hist3, past_trials).mean(-1) * (1 - inv_mask[:,:,3].unsqueeze(-1)) * past_trial_lengths_binary.squeeze(-1)).view(-1, self.trial_seq_len), past_trial_lengths_safe.flatten(), batch_first=True, enforce_sorted=False).data)

    loss = static_loss1 + dx_loss1 + rx_loss1 + history_loss1 + static_loss2 + dx_loss2 + rx_loss2 + history_loss2 + static_loss3 + dx_loss3 + rx_loss3 + history_loss3
    return loss

  def forward(self, trial, investigators, inv_dx, inv_dx_lens, inv_rx, inv_rx_lens, past_trials, past_trials_lengths, inv_mask, num_inv):
    # Trial is (bs, trial_dim)
    # All other inputs are (bs, M, *) where * is either a sequence or single input
    with torch.no_grad():
      self.imputer.eval()
      _, (investigators, inv_dx, inv_rx, past_trials) = self.imputer(investigators, inv_dx, inv_dx_lens, inv_rx, inv_rx_lens, past_trials, past_trials_lengths, inv_mask, num_inv)

    investigator_encoding = self.static_encoder(investigators)
    dx_encoding = self.diagnosis_encoder(inv_dx, inv_dx_lens, num_inv)
    rx_encoding = self.prescription_encoder(inv_rx, inv_rx_lens, num_inv)
    history_encoding = self.history_encoder(past_trials, past_trials_lengths, num_inv)
    trial_encoding = self.trial_encoder(trial)
    trial_encoding = trial_encoding.unsqueeze(1).repeat(1, num_inv, 1)
    inv_site_representation = self.inv_encoder(torch.cat((trial_encoding, investigator_encoding, dx_encoding, rx_encoding, history_encoding), dim=-1))
    network_input = self.transformer_encoder(inv_site_representation.transpose(0,1)).transpose(0,1)
    score = self.output(torch.relu(self.fc(network_input)))
    return score



class MCATFullData(nn.Module):
  def __init__(self, trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers, hidden_dim, n_heads):
    super(MCATFullData, self).__init__()
    self.static_encoder = nn.Linear(static_dim, embedding_dim)
    self.diagnosis_encoder = DiagnosisEncoder(dx_dim, lstm_dim, embedding_dim, num_layers)
    self.prescription_encoder = PrescriptionEncoder(rx_dim, lstm_dim, embedding_dim, num_layers)
    self.history_encoder = PastTrialEncoder(trial_dim, lstm_dim, embedding_dim, num_layers)
    self.trial_encoder = nn.Linear(trial_dim, embedding_dim)
    self.inv_encoder = nn.Linear(5 * embedding_dim, 2 * embedding_dim)
    
    encoder_layer = nn.TransformerEncoderLayer(2*embedding_dim, n_heads, hidden_dim)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 2)
    self.fc = nn.Linear(2*embedding_dim, hidden_dim)
    self.output = nn.Linear(hidden_dim, 1)

  def forward(self, trial, investigators, inv_dx, inv_dx_lens, inv_rx, inv_rx_lens, past_trials, past_trials_lengths, num_inv):
    # Trial is (bs, trial_dim)
    # All other inputs are (bs, M, *) where * is either a sequence or single input
    investigator_encoding = self.static_encoder(investigators)
    dx_encoding = self.diagnosis_encoder(inv_dx, inv_dx_lens, num_inv)
    rx_encoding = self.prescription_encoder(inv_rx, inv_rx_lens, num_inv)
    history_encoding = self.history_encoder(past_trials, past_trials_lengths, num_inv)
    trial_encoding = self.trial_encoder(trial)
    trial_encoding = trial_encoding.unsqueeze(1).repeat(1, num_inv, 1)
    inv_site_representation = self.inv_encoder(torch.cat((trial_encoding, investigator_encoding, dx_encoding, rx_encoding, history_encoding), dim=-1))
    network_input = self.transformer_encoder(inv_site_representation.transpose(0,1)).transpose(0,1)
    score = self.output(torch.relu(self.fc(network_input)))
    return score



#######
### Synthetic Data Labeler
#######

class DataLabeler(nn.Module):
  def __init__(self, trial_dim, static_dim, dx_dim, rx_dim, lstm_dim, embedding_dim, num_layers, hidden_dim):
    super(DataLabeler, self).__init__()
    self.static_encoder = nn.Linear(static_dim, embedding_dim)
    self.diagnosis_encoder = DiagnosisEncoder(dx_dim, lstm_dim, embedding_dim, num_layers)
    self.prescription_encoder = PrescriptionEncoder(rx_dim, lstm_dim, embedding_dim, num_layers)
    self.history_encoder = PastTrialEncoder(trial_dim, lstm_dim, embedding_dim, num_layers)
    self.trial_encoder = nn.Linear(trial_dim, embedding_dim)
    self.inv_encoder = nn.Linear(5 * embedding_dim, 2 * embedding_dim)
    self.fc = nn.Linear(2*embedding_dim, hidden_dim)
    self.output = nn.Linear(hidden_dim, 1)

  def forward(self, trial, investigators, inv_dx, inv_dx_lens, inv_rx, inv_rx_lens, past_trials, past_trials_lengths, num_inv):
    # Trial is (bs, trial_dim)
    # All other inputs are (bs, M, *) where * is either a sequence or single input
    investigator_encoding = torch.relu(self.static_encoder(investigators))
    dx_encoding = torch.relu(self.diagnosis_encoder(inv_dx, inv_dx_lens, num_inv))
    rx_encoding = torch.relu(self.prescription_encoder(inv_rx, inv_rx_lens, num_inv))
    history_encoding = torch.relu(self.history_encoder(past_trials, past_trials_lengths, num_inv))
    trial_encoding = torch.relu(self.trial_encoder(trial))
    trial_encoding = trial_encoding.unsqueeze(1).repeat(1, num_inv, 1)
    inv_site_representation = torch.relu(self.inv_encoder(torch.cat((trial_encoding, investigator_encoding, dx_encoding, rx_encoding, history_encoding), dim=-1)))
    enrollment = torch.relu(self.output(torch.relu(self.fc(inv_site_representation))))
    return enrollment