import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import SurvivalData
import log_rank_test
import random
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error as mae
import dataset_demo
import torch.nn.functional as F
import re

##Functions to sort folder, files in the "natural" way:
def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


# define the true objective function
def objective(x, a, b):
    return np.exp(-(x / b) ** a)

def post_process(real_labels, filter_limit=False, filter_correction=False):

    processed_labels = []
    if filter_limit:
        living_worms_first_day = max(real_labels)
        for lab in range(len(real_labels)):
            processed_labels.append(min(real_labels[lab], living_worms_first_day))
    if filter_correction:
        for lab2 in range(len(real_labels)):
            x = processed_labels[lab2]
            i = 1
            while lab2-i >= 0 and x > processed_labels[lab2-i]:
                processed_labels[lab2-i] = x
                i += 1
    p = [int(round(a)) for a in processed_labels]
    return p

def count_to_test_format(count, init_individuals, day_init, name):
    days = []
    deads = []

    for i in range(len(count)):
        count_today = count[i]
        previous_count = count[i - 1]

        if i == 0:
            deads_today = init_individuals - count_today

        else:
            deads_today = previous_count - count_today

        deads.append(deads_today)
        days.append(day_init + i)

        if count_today == 0:
            break

    test_format = '%' + name + '[' + str(init_individuals) + ']'
    test_format += '\n#days\tdead\tcensored'

    for j in range(len(deads)):
        test_format += '\n' + str(days[j]) + '\t' + str(deads[j])
    return test_format

# Python3 program to check if
# all elements in a list are identical
def detect_constant_curves(list):
    return all(i == list[0] for i in list)

def detect_constant_curves_noise(list, tolerance):
    if list[-1] >= list[0] - tolerance and list[-1] <= list[0] + tolerance:
        return True
    else:
        return False


class PadSequence_test_orig:
    def __call__(self, batch, max_len=57):
        images = []
        counts = []
        labels = []
        subdirs = []
        nseqs = len(batch)
        seq_lens = []
        labels_lens = []

        for i in range(nseqs):
            l = len(batch[i][0])
            seq_lens.append(l)
            ll = len(batch[i][2])
            labels_lens.append(ll)
            a = batch[i][0]
            images.append(F.pad(a, (0, 0, 0, 0, 0, 0, 0, (max_len - l))))
            b = batch[i][1]
            counts.append(F.pad(b, (0, (max_len - l))))
            c = batch[i][2]
            labels.append(F.pad(c, (0, (max_len - ll))))
            subdirs.append(batch[i][3])

        return images, counts, labels, seq_lens, labels_lens, subdirs


def generate_curve_weibull_est(day_init, n_objects, stepness, mean_life):
    alive = n_objects
    curve = []
    days = []
    t = day_init

    while alive > 0:
        survival = np.exp(-(t / mean_life) ** stepness)
        alive = int(round((survival*n_objects)))
        curve.append(alive)
        days.append(t)
        t += 1

        if t > 100:
            break

    dead_curve = [n_objects - x for x in curve]
    return curve, dead_curve

def predict_lifespan_curve(conddir, gaps_days, idx_ini, nw, device):

    exp_name = conddir.split('/')[-1]
    data_transform = transforms.ToTensor()
    print('--------------------------------------------------------------')
    print(exp_name)
    print('--------------------------------------------------------------')

    plates = os.listdir(conddir)
    plates.sort(key=natural_keys)

    batch_size = 1
    array_length = 100
    seq_length = 57

    for idx_stop in range(idx_ini, 45):
        print('\n--> Number of inputs: ' + str(idx_stop))

        condition_count = np.zeros((batch_size, array_length))
        condition_manual_count = np.zeros((batch_size, array_length))

        condition_model_count = np.zeros(array_length)
        condition_weibull_count = np.zeros(array_length)

        std_list = []

        for plate in plates:
            testdir = conddir + '/' + plate
            print(testdir)

            ## Load data
            data_test = dataset_demo.LifespanDatasetTest(root_dir=testdir, seq_length=57,transform=data_transform,index_stop=idx_stop)
            dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False, num_workers=0,collate_fn=PadSequence_test_orig())

            iters = 100 # Number of samples used for uncertainty estimation

            with torch.no_grad():

                batch = next(iter(dataloader_test))
                imgs_batch = torch.stack(batch[0]).to(device) # Images captured up to the current moment
                past_counts_batch = torch.stack(batch[1]).to(device) # Counts up to the present day
                labels = torch.stack(batch[2]).to(device) # Future days labels
                seq_lens = torch.Tensor(batch[3]) # Longitud de la secuencia de entrada

                preds_count = np.zeros((batch_size, iters, seq_length))
                preds_count_weibull =  np.zeros((batch_size, iters, 57))

                inputs_list = past_counts_batch.tolist()[0]
                finished_plate = 0
                if inputs_list[int(seq_lens[0].cpu().numpy())-1] == 0:
                    finished_plate = 1

                inputs_list = [inp for inp in inputs_list if inp != 0]
                constant_case = 0

                for it in range(iters):
                    if detect_constant_curves(inputs_list):
                        constant_case = 1

                    else:
                        if it != 0:
                            # Normal random noise of mean 0 variance 1 is added to the input.
                            noise = torch.randn(int(seq_lens[0].cpu().numpy())).to(device)
                            zeros = torch.zeros(seq_length - int(seq_lens[0].cpu().numpy())).to(device)
                            inpt_noise = torch.cat((noise, zeros), dim=0)
                            inpt_noise = torch.unsqueeze(inpt_noise, 0)
                            inpt = torch.add(past_counts_batch, inpt_noise)
                        else:
                            inpt = past_counts_batch

                        ### to detect whether it is a case of constant
                        inputs_list = inpt.tolist()[0]
                        inputs_list = [inp for inp in inputs_list if inp != 0]

                    if detect_constant_curves_noise(inputs_list,tolerance=0.5) or constant_case == 1:
                        mu_ml, sigma_ml = 29.37, 14.8
                        lower_ml, upper_ml = len(inputs_list), mu_ml + 2 * sigma_ml

                        mu_st, sigma_st = 4.64, 2.16
                        lower_st, upper_st = 4, mu_st + 2 * sigma_st

                        if random.getrandbits(1) == 0:
                            sample_mean_life = upper_ml
                            sample_stepness = upper_st
                        else:
                            sample_mean_life = lower_ml
                            sample_stepness = lower_st


                        pred1, _ = generate_curve_weibull_est(day_init=len(inputs_list), n_objects=inputs_list[0], stepness=abs(sample_stepness), mean_life=abs(sample_mean_life))

                        if len(pred1) > seq_length:
                            pred1 = pred1[0:seq_length]

                        if len(pred1) < seq_length:
                            pred1.extend([0] * (seq_length - len(pred1)))

                        for b in range(batch_size):
                            preds_count[b, it, :] = pred1
                            preds_count_weibull[b, it, :] = pred1

                    elif finished_plate == 1:
                        pred1 = [0] * seq_length
                        for b in range(batch_size):
                            preds_count[b, it, :] = pred1
                            preds_count_weibull[b, it, :] = pred1


                    else: # Predict with the NN model
                        pre = nw(imgs_batch, inpt, seq_lens)

                        list_inpts = inpt.tolist()[0]

                        celeg_ini = list_inpts[0]
                        list_inpts_per = []
                        for c in list_inpts:
                            list_inpts_per.append(c / celeg_ini)

                        list_inpts_per = list_inpts_per[0:idx_stop]

                        try:
                            p0 = np.array([3, 25])
                            x = list(range(4, 4 + len(list_inpts_per)))
                            popt, _ = curve_fit(objective, x, list_inpts_per, p0, maxfev=10000)
                            a, b = popt
                            curve_est, death_curve = generate_curve_weibull_est(4, celeg_ini, a, b)
                            curve_est.extend([0] * (seq_length - len(curve_est)))
                        except Exception as e:
                            print(e)

                            mu_ml, sigma_ml = 29.37, 14.8
                            lower_ml, upper_ml = len(list_inpts), mu_ml + 2 * sigma_ml
                            mu_st, sigma_st = 4.64, 2.16
                            lower_st, upper_st = 4, mu_st + 2 * sigma_st

                            if random.getrandbits(1) == 0:
                                sample_mean_life = upper_ml
                                sample_stepness = upper_st
                            else:
                                sample_mean_life = lower_ml
                                sample_stepness = lower_st

                            # a Weibull curve is generated with the sampled parameters
                            curve_est, _ = generate_curve_weibull_est(day_init=len(list_inpts),
                                                                        n_objects=list_inpts[0],
                                                                        stepness=abs(sample_stepness),
                                                                        mean_life=abs(sample_mean_life))

                        ## adapt the curve to the sequence length
                        if len(curve_est) > seq_length:
                            curve_est = curve_est[0:seq_length]

                        if len(curve_est) < seq_length:
                            curve_est.extend([0] * (seq_length - len(curve_est)))

                        for b in range(batch_size):
                            post_pred = post_process(pre[b].cpu().numpy(), filter_limit=True, filter_correction=True)

                            ## adapt to gap days
                            for p in range(idx_stop + 4, 61):
                                if p in gaps_days:
                                    if p - (idx_stop + 4) != 0:
                                        post_pred[p - (idx_stop + 4)] = post_pred[p - (idx_stop + 5)]

                            preds_count[b, it, :] = post_pred
                            preds_count_weibull[b, it, :] = curve_est


                # calculate the mean and variance of the predictions
                preds_count_mean = preds_count.mean(axis=1)
                preds_count_std = preds_count.std(axis=1)
                std_list.append(preds_count_std[0])

                ####### update condition mean model count
                plate_model_count = preds_count[0][0]
                aux = np.zeros(array_length - len(plate_model_count))
                plate_model_count = np.concatenate((plate_model_count, aux), axis=None)
                condition_model_count = np.add(condition_model_count, plate_model_count)
                ####### update condition weibull count
                plate_weibull_count = preds_count_weibull[0][0]
                aux = np.zeros(array_length - len(plate_weibull_count))
                plate_weibull_count = np.concatenate((plate_weibull_count, aux), axis=None)
                condition_weibull_count = np.add(condition_weibull_count, plate_weibull_count)
                ####### update condition manual count
                plate_manual_count = np.concatenate((past_counts_batch[0].cpu().numpy()[0:int(seq_lens[0].cpu().numpy())], labels[0].cpu().numpy()), axis=None)
                aux = np.zeros((1, array_length-len(plate_manual_count)))
                plate_manual_count = np.concatenate((plate_manual_count, aux), axis=None)
                plate_manual_count = plate_manual_count.reshape(1,len(plate_manual_count))
                condition_manual_count = np.add(condition_manual_count, plate_manual_count)
                #### update condition mean model count
                aux = np.zeros((1, array_length - len(preds_count_mean[0])))
                preds_count_mean = np.concatenate((preds_count_mean[0], aux), axis=None)
                condition_count = np.add(condition_count,preds_count_mean)

        ### Condition analysis ####

        upper_lim_cond = []
        lower_lim_cond = []

        ### law of error propagation
        for i in range(len(condition_manual_count[0])):
            day_variability = 0
            if i < 57:
                for j in range(len(std_list)):
                    var = pow(std_list[j][i], 2)
                    day_variability += var
                day_variability =  np.sqrt(day_variability)

            else:
                day_variability = 0

            z_alfa = 1.96
            samples = len(plates)

            upper_lim_cond.append(condition_count[0][i] + z_alfa * day_variability / np.sqrt(samples))
            lower_lim_cond.append(condition_count[0][i] - z_alfa * day_variability / np.sqrt(samples))


        #### Graph of count by condition #######
        plt.figure(figsize=(1855 / 96, 986 / 96), dpi=96)
        plt.title('Exp: ' + exp_name + " Input sequence lenght: " + str(int(seq_lens[0].cpu().numpy())))
        days_prev = np.arange(4, 4 + seq_lens[0].cpu().numpy())
        counts_prev = condition_manual_count[0][0:int(seq_lens[0].cpu().numpy())]
        plt.plot(days_prev, counts_prev, "y-", label="previous days", linewidth=3)
        days_fut = np.arange(4 + seq_lens[0].cpu().numpy(), 4 + seq_lens[0].cpu().numpy() + len(upper_lim_cond) )
        plt.plot(days_fut, condition_count[0],  "b-+", label="Model_mean", linewidth=3)
        plt.plot(days_fut, condition_model_count,  "r-+", label="Model", linewidth=3)
        days = np.arange(4, array_length+4)
        plt.plot(days_fut[0:array_length - int(seq_lens[0].cpu().numpy())], condition_manual_count[0][int(seq_lens[0].cpu().numpy()):],  "g-", label="Real", linewidth=3)
        plt.plot(days, condition_weibull_count,  "c-+", label="Weibull", linewidth=3)
        plt.fill_between(days_fut, lower_lim_cond, upper_lim_cond, alpha=.1,color='blue')
        plt.axhline(y=int(condition_manual_count[0][0]/2), color='r', linestyle='--')
        plt.axvline(x=idx_stop+3, color='b',linestyle='--')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel(" Days", fontsize=15, fontweight='bold')
        plt.ylabel("Live C. elegans", fontsize=15, fontweight='bold')
        plt.grid(True, axis="y")
        plt.legend(fontsize=20)
        plt.show()
        plt.close()

        ### LOG RANK TEST ####
        counts_prev = list(map(int, counts_prev))
        upper_lim_cond = [int(round(num)) for num in upper_lim_cond]
        lower_lim_cond = [int(round(num)) for num in lower_lim_cond]
        init_indiv = int(condition_manual_count[0][0])

        upper_lim_cond = counts_prev + upper_lim_cond
        lower_lim_cond = counts_prev + lower_lim_cond

        test_values = count_to_test_format(upper_lim_cond, init_indiv, 4, 'upper')
        test_values2 = count_to_test_format(lower_lim_cond, init_indiv, 4, 'lower')

        testTxt = test_values + '\n\n' + test_values2
        conditions = []
        SurvivalData.ReadData( testTxt.split( "\n" ), conditions)
        results_logRankTest = log_rank_test.LogRankTestAll(conditions)

        print('\nLog rank test results')

        print('\n--------------------------------------------------------------------------------------------------------------')
        print("{:<12} {:<12} {:<30} {:<30} {:<10}".format('Sample 1', 'Sample 2', 'Chi^2' , 'P-value', 'Corrected P-value'))
        print('--------------------------------------------------------------------------------------------------------------')

        for v in results_logRankTest:
            samp1, samp2, chi, pv, corrpv = v
            print("{:<12} {:<12} {:<30} {:<30} {:<10}".format(samp1, samp2, chi, pv, corrpv))
            print('--------------------------------------------------------------------------------------------------------------')

        alfa = 0.05
        p_value = results_logRankTest[0][3]

        if p_value > alfa:
            print('P-value:  ' + str(round(p_value, 4)) + ' > alpha=' + str(alfa) + ' \ntherefore the '
            'the null hypothesis is accepted and it is concluded that there are no significant differences between the extremes of the confidence interval')
            print('End of the assay')

            count_weib_perc = []
            count_manual_perc = []
            count_nn_perc = []

            future_real_cnts = condition_manual_count[0][idx_stop:]
            future_weib_cnts = condition_weibull_count[idx_stop:]

            ini_celeg = condition_manual_count[0][0]

            # convert counts to survival percentage
            for c in range(len(future_real_cnts)):
                count_manual_perc.append(future_real_cnts[c]/ini_celeg)
                count_weib_perc.append(future_weib_cnts[c]/ini_celeg)
                count_nn_perc.append(condition_model_count[c]/ini_celeg)

            dur1 = next(d for d in range(len(count_manual_perc)) if count_manual_perc[d] == 0)  # day when no one is left alive
            dur2 = next(d for d in range(len(count_nn_perc)) if count_nn_perc[d] == 0)
            dur3 = next(d for d in range(len(count_weib_perc)) if count_weib_perc[d] == 0)

            # Calculate mae
            mae_nn = mae(count_manual_perc[: max(dur1, dur2) + 1], count_nn_perc[: max(dur1, dur2) + 1])
            mae_weib = mae(count_manual_perc[: max(dur1, dur3) + 1], count_weib_perc[: max(dur1, dur3) + 1])

            count_manual_logrank_format = count_to_test_format(condition_manual_count[0], ini_celeg, 4, 'manual')
            cnt_nn = [int(round(num)) for num in condition_model_count]

            cnt_nn = counts_prev + cnt_nn
            count_nn_logrank_format = count_to_test_format(cnt_nn, ini_celeg, 4, 'NN')

            testTxt2 = count_manual_logrank_format + '\n\n' + count_nn_logrank_format
            conditions = []
            SurvivalData.ReadData(testTxt2.split("\n"), conditions)
            results_logRankTest2 = log_rank_test.LogRankTestAll(conditions)

            p_value2 = results_logRankTest2[0][3]

            print('\nResults')
            print('\n-----------------------------------------------------------------')
            print("{:<12} {:<20} {:<10} {:<30}".format('Day stop','MAE_Weibull(%)', 'MAE(%)', 'P-value'))
            print('-----------------------------------------------------------------')
            print("{:<12} {:<20} {:<10} {:<30}".format((idx_stop + 3), round(mae_weib * 100, 2), round(mae_nn * 100, 2), p_value2))
            print('-----------------------------------------------------------------')
            break

        else:
            print('P-value:  ' + str(round(p_value, 4)) + ' < alpha=' + str(alfa) + '\ntherefore the '
             'the null hypothesis is rejected and it is concluded that there are significant differences between the extremes of the confidence interval')
