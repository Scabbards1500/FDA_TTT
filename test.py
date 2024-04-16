# if memory_size > buffer_size:
#     with torch.no_grad():
#         #
#         # retrieved_batches = memory.get_neighbours(features.cpu().numpy(), k=4)  # 4
#         # pseudo_past_logits = retrieved_batches.cuda()
#         # pseudo_current_logits = outputs
#         #
#         pseudo_current_logits = outputs  # 本次的mask
#         retrieved_batches = memory.get_neighbours(style.cpu().numpy(), k=4)  # 找最接近的几个风格
#         pseudo_past_style = retrieved_batches.cuda()  # 找的近似的(1,3,256,256)
#         pseudo_past_logits_input = arc_add_amp(pseudo_past_style, amp, pha, L=0.5)  # 更改过去风格后的本次图片
#         pseudo_past_logits = model(pseudo_past_logits_input)  # 更改过去风格后的本次图片得到的mask
#         pseudo_past_labels = nn.functional.softmax(pseudo_past_logits, dim=1)  # 近似的softmax
#         pseudo_current_labels = nn.functional.softmax(pseudo_current_logits / 2, dim=1)  # 本次输出的softmax
#         diff_loss = (F.kl_div(pseudo_past_labels.log(), pseudo_current_labels, None, None, 'none') + F.kl_div(
#             pseudo_current_labels.log(), pseudo_past_labels, None, None, 'none')) / 2  # kl散度
#         diff_loss = torch.sum(diff_loss, dim=1)  # 获取的loss
#         diff_loss = diff_loss.cpu().numpy().tolist()
#         sum_loss = sum(sum(sublist) for sublist in diff_loss[0])
#         len_loss = len(diff_loss[0])
#         diff_loss = sum_loss / len_loss
#
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = diff_loss * 1e-6





            # diff_loss = (F.kl_div(pseudo_past_style.log(), style, None, None, 'none') + F.kl_div(
            #     style.log(), pseudo_past_style, None, None, 'none')) / 2  #kl散度
            # diff_loss = torch.sum(diff_loss, dim=1) # 获取的loss
            # diff_loss = diff_loss.cpu().numpy().tolist()
            # sum_loss= sum(sum(sublist) for sublist in diff_loss[0])
            # len_loss= len(diff_loss[0])
            # diff_loss = (sum_loss / len_loss)* 1e-6

            # if memory_size > buffer_size:
            #     with torch.no_grad():
            #         pseudo_current_logits = outputs  # 本次的mask
            #         retrieved_batches = memory.get_neighbours(style.cpu().numpy(), k=4)  # 找最接近的几个风格
            #         pseudo_past_style = retrieved_batches.cuda() # 找的近似的(1,3,256,256)
            #         pseudo_past_logits_input = arc_add_amp(pseudo_past_style, amp,pha,L=0.8) #更改过去风格后的本次图片
            #         pseudo_past_logits = model(pseudo_past_logits_input) # 更改过去风格后的本次图片得到的mask
            #         pseudo_past_labels = nn.functional.softmax(pseudo_past_logits, dim=1) # 近似的softmax,这个softmax按道理应该换成dice
            #         pseudo_current_labels = nn.functional.softmax(pseudo_current_logits / 2, dim=1) # 本次输出的softmax
            #         diff_loss = (F.kl_div(pseudo_past_labels.log(), pseudo_current_labels, None, None, 'none') + F.kl_div(
            #             pseudo_current_labels.log(), pseudo_past_labels, None, None, 'none')) / 2  #kl散度
            #         diff_loss = torch.sum(diff_loss, dim=1) # 获取的loss
            #         diff_loss = diff_loss.cpu().numpy().tolist()
            #         sum_loss= sum(sum(sublist) for sublist in diff_loss[0])
            #         len_loss= len(diff_loss[0])
            #         diff_loss = sum_loss / len_loss

print("helloworld")