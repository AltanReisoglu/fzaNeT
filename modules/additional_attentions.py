import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformableAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, kernel_size, stride=1, padding=0, bias=False):
        super(DeformableAttention, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding


        self.head_dim = out_channels // num_heads
        assert self.head_dim * num_heads == out_channels, "out_channels, num_heads'e tam bölünmelidir."

        # Query, Key, Value projeksiyonları
        # input_features'dan Q, K, V'yi türeteceğiz
        self.query_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.key_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        # Offset ve Attention Ağırlıkları (Maske) için projeksiyonlar
        # Offsetleri öğrenmek için bir evrişim katmanı
        # Her bir kafa için kernel_size * kernel_size * 2 (x, y koordinatları) offset tahmin ediyoruz.
        self.offset_conv = nn.Conv2d(in_channels, 
                                     num_heads * kernel_size * kernel_size * 2, # 2 for (x,y) offset
                                     kernel_size=kernel_size, 
                                     stride=stride, 
                                     padding=padding, 
                                     bias=True) # Offsetler için bias genellikle iyi sonuç verir.
        
        # Dikkat ağırlıklarını (maskelerini) öğrenmek için bir evrişim katmanı
        # Her bir kafa için kernel_size * kernel_size dikkat ağırlığı (0-1 arası) tahmin ediyoruz.
        self.attention_weights_conv = nn.Conv2d(in_channels, 
                                                num_heads * kernel_size * kernel_size, 
                                                kernel_size=kernel_size, 
                                                stride=stride, 
                                                padding=padding, 
                                                bias=True)

        # Çıkış projeksiyonu
        self.out_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, input_features):
        N, C, H, W = input_features.shape # Batch, Channels, Height, Width

        # Q, K, V Projeksiyonları
        query = self.query_proj(input_features) # (N, out_channels, H, W)
        key = self.key_proj(input_features)     # (N, out_channels, H, W)
        value = self.value_proj(input_features) # (N, out_channels, H, W)

        # Offsetleri ve Dikkat Ağırlıklarını Öğrenme
        # offsets: (N, num_heads * k_h * k_w * 2, H_out, W_out)
        offsets = self.offset_conv(input_features)
        
        # attention_weights: (N, num_heads * k_h * k_w, H_out, W_out)
        # Sigmoid kullanarak ağırlıkları 0-1 arasına sıkıştırıyoruz
        attention_weights = torch.sigmoid(self.attention_weights_conv(input_features))
        
        output_h = offsets.shape[2]
        output_w = offsets.shape[3]

        # Örnekleme Izgarası Oluşturma (Grid Generation)
        # Normalizasyon için [ -1, 1 ] aralığında ızgara oluştur.
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, output_h, device=input_features.device),
                                        torch.linspace(-1, 1, output_w, device=input_features.device),
                                        indexing='ij')
        
        grid = torch.stack((grid_x, grid_y), dim=-1) # (H_out, W_out, 2)
        grid = grid.unsqueeze(0).expand(N, -1, -1, -1) # (N, H_out, W_out, 2)

        # Çıktı başlıklarının birleştirileceği boş liste
        output_features_per_head = []

        # Her bir dikkat başlığı için işlem
        for i_head in range(self.num_heads):
            # Başlığa özel Q, K, V dilimlerini al
            head_query = query[:, i_head * self.head_dim : (i_head + 1) * self.head_dim, :, :]
            head_key = key[:, i_head * self.head_dim : (i_head + 1) * self.head_dim, :, :]
            head_value = value[:, i_head * self.head_dim : (i_head + 1) * self.head_dim, :, :]

            # Başlığa özel ofsetleri ve dikkat ağırlıklarını al
            # offsets: (N, num_heads * k*k * 2, H_out, W_out) -> (N, k*k, 2, H_out, W_out)
            head_offsets = offsets[:, i_head * self.kernel_size * self.kernel_size * 2 : 
                                      (i_head + 1) * self.kernel_size * self.kernel_size * 2, :, :]
            head_offsets = head_offsets.view(N, self.kernel_size * self.kernel_size, 2, output_h, output_w)
            head_offsets = head_offsets.permute(0, 3, 4, 1, 2) # (N, H_out, W_out, k*k, 2)

            # attention_weights: (N, num_heads * k*k, H_out, W_out) -> (N, k*k, H_out, W_out)
            head_attention_weights = attention_weights[:, i_head * self.kernel_size * self.kernel_size : 
                                                          (i_head + 1) * self.kernel_size * self.kernel_size, :, :]
            head_attention_weights = head_attention_weights.permute(0, 2, 3, 1) # (N, H_out, W_out, k*k)
            
            sampled_values_for_head = []
            for i_point in range(self.kernel_size * self.kernel_size):
                # Her bir örnekleme noktası için ofset ve ağırlık
                point_offset = head_offsets[:, :, :, i_point, :] # (N, H_out, W_out, 2)
                point_attention_weight = head_attention_weights[:, :, :, i_point].unsqueeze(-1) # (N, H_out, W_out, 1)

                # Ofsetleri normalize edilmiş ızgaraya ekle
                # Burada ofsetlerin piksel biriminden olduğunu varsayarak [-1, 1] aralığına ölçeklememiz gerekir.
                # Daha doğru bir yaklaşım için, input_height ve input_width kullanarak normalize etmeliyiz.
                # Bu örnekte basitleştirilmiş bir sabitle ölçekleme yapalım:
                # Ofsetleri küçük bir faktörle ölçekleyelim, böylece ızgaradan çok dışarı çıkmasınlar.
                # Daha gerçekçi bir Deformable Conv'da ofsetler piksel cinsinden olur ve sonra normalize edilir.
                # Örn: scaled_offsets = point_offset / torch.tensor([W/2, H/2], device=input_features.device)
                
                # Geçici olarak ofsetleri küçük bir faktörle ölçekleyelim
                scaled_offsets = point_offset * (2.0 / torch.tensor([W, H], device=input_features.device)) # Örneğin 2.0 / Genişlik veya Yükseklik

                sampling_grid_points = grid + scaled_offsets 

                # Bilinear interpolasyon ile değerleri örnekle
                # F.grid_sample (N, C, H_in, W_in) ve (N, H_out, W_out, 2) grid bekler.
                sampled_value = F.grid_sample(head_value, sampling_grid_points, 
                                              mode='bilinear', padding_mode='zeros', align_corners=True)
                
                # Örneklenen değeri dikkat ağırlığı ile çarp
                # sampled_value: (N, head_dim, H_out, W_out)
                # point_attention_weight: (N, H_out, W_out, 1) -> (N, 1, H_out, W_out) for broadcasting
                weighted_sampled_value = sampled_value * point_attention_weight.permute(0, 3, 1, 2)
                
                sampled_values_for_head.append(weighted_sampled_value)
            
            # Tüm kernel noktalarından örneklenen ağırlıklı özellikleri topla
            head_output = torch.sum(torch.stack(sampled_values_for_head, dim=0), dim=0)
            output_features_per_head.append(head_output)

        # Başlıkların çıktılarını birleştir
        combined_output = torch.cat(output_features_per_head, dim=1) # (N, out_channels, H_out, W_out)
        
        # Son çıkış projeksiyonu
        output = self.out_proj(combined_output)

        return output

# --- Kullanım Örneği ---
if __name__ == "__main__":
    # Test parametreleri
    batch_size = 1
    in_channels = 256
    out_channels = 1
    num_heads = 1
    kernel_size = 3 # Deformable Conv benzeri kernel boyutu
    input_height = 56
    input_width = 56

    # Sahte girdi özelliği haritası
    input_features = torch.randn(batch_size, in_channels, input_height, input_width)

    # Deformable Attention katmanı oluşturma
    # padding, output boyutunu inputla aynı tutmak için kernel_size // 2 olmalı
    deform_attn_qkv = DeformableAttention(in_channels, out_channels, num_heads, 
                                                 kernel_size, padding=kernel_size // 2) 

    # İleri besleme (forward pass)
    output_qkv = deform_attn_qkv(input_features)

    print(f"Girdi boyutu: {input_features.shape}")
    print(f"Çıktı boyutu: {output_qkv.shape}")

    # Küçük bir test daha
    assert output_qkv.shape == (batch_size, out_channels, input_height, input_width), "Çıktı boyutu yanlış!"

    print("\nQKV projeksiyonlu Deformable Attention katmanı başarıyla çalıştı!")

    # Farklı kernel boyutu ile deneme
    deform_attn_qkv_k5 = DeformableAttention(in_channels, out_channels, num_heads, 
                                                    kernel_size=5, padding=5 // 2)
    output_qkv_k5 = deform_attn_qkv_k5(input_features)
    print(f"\nKernel boyutu 5 için çıktı boyutu: {output_qkv_k5.shape}")
    assert output_qkv_k5.shape == (batch_size, out_channels, input_height, input_width), "Çıktı boyutu yanlış!"