in_feature = torch.randn(69,240,320)
kernel = torch.randn(1,69,3,3)
out_channel = kernel.size(0)

in_channels = in_feature.size(0)
orig_h, orig_w = in_feature.size(1), in_feature.size(2)

#Kernel Dimenstions
kh, kw = kernel.size(2), kernel.size(3)
#Strides
dh, dw = 1, 1

#Padding --> o = [i+2p-k/s]+1 && o = i
p = int((kh-1)/2)
img = F.pad(input= in_feature, pad= (p, p, p, p), mode='constant', value= 0)

out_h = (orig_h+2*p-kh) + 1
out_w = (orig_w+2*p-kw) + 1

#Image Dimenstions
h, w = img.size(1), img.size(2)

#Creating the patches - over which convolution is done
patches = img.unfold(1, kh, dh).unfold(2, kw, dw).reshape(-1, in_channels, kh, kw)
#To parallelize the operation
#[b,L,c,kh,kw] --> [b,L,c*kh*kw]
patches = patches.reshape(patches.size(0), -1) 

#Reshaping the kernel for parallelization
#[o,c,kh,kw] --> [o, c*kh*kw]
k = kernel.reshape(out_channel, -1) 
#result = torch.zeros(batch_size, out_channel, orig_h, orig_w)

patches= patches.type(torch.cuda.FloatTensor)

#Convolution Operation
#Actually it cross-correlation that is carried out!... 
#x is a float val that is inserted in the appropriate position in output tensor --> result
o = out_channel
L = patches.size(0)

n_conv = L*o
n_mul = L*o*patches.size(1)
# for o in range(out_channel):
# 	for L in range(patches.size(0)):
# 		x = convolve(patches[L], k[o])
# 		result[b][o][L//orig_h][L%orig_w] = x

print(n_conv) 			#No of conv
print(n_mul)			#No of mul
print(out_h, out_w)		#Output dimensions