import torch
import time


def train_fourier_2D(model, training_set, testing_set, epochs, learning_rate, validation_set=None, wd=1e-5, l=torch.nn.MSELoss()):
    """ This function train a model. It performs 'epochs' epochs, the optimizer is Adam('learning_rate') and 
    there is a learning rate optimizer"""
    
    N_batch = len(training_set)

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=wd,lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, epochs=epochs, steps_per_epoch=N_batch)
   
    start_time_training = time.time()
    avg_time = 0.
    current_training_time = 0.

    historic_rel_train = []
    historic_loss_train = []

    historic_rel_test = []
    historic_loss_test = []

    historic_rel_val = []
    historic_loss_val = []

    historic_lr = []

    for epoch in range(epochs):

        start_time_epoch = time.time()
        model.train()
        train_err = 0.
        rel_train_err = 0.

        for i_t, (input_batch, output_batch) in enumerate(training_set):

            optimizer.zero_grad()

            loss_f = 0.
            output_pred_batch = model(input_batch)
            loss_f = l(output_pred_batch, output_batch).float()

            train_err += loss_f.item()
            rel_train_err += torch.mean(torch.sqrt(torch.sum((output_pred_batch-output_batch)**2, axis=(1,2,3))/torch.sum(output_batch**2, axis=(1,2,3)))).item()

            # Compute gradients
            loss_f.backward()

            # Update the parameters
            optimizer.step()
            scheduler.step()

        print("Boucle faite")
        train_err /= len(training_set)
        rel_train_err /= len(training_set)

        print("Erreur calculée")
        historic_lr.append(optimizer.param_groups[0]['lr']) 
        historic_rel_train.append(rel_train_err)
        historic_loss_train.append(train_err)
        print("loss ajoutée")

        # Compute test loss (it follows the same steps as above)
        with torch.no_grad():

            model.eval()
            test_err = 0.
            rel_test_err = 0.

            for i_t, (input_batch, output_batch) in enumerate(testing_set):

                loss_f = 0.
                output_pred_batch = model(input_batch)
                loss_f = l(output_pred_batch, output_batch).float()

                test_err += loss_f.item()
                rel_test_err += torch.mean(torch.sqrt(torch.sum((output_pred_batch-output_batch)**2, axis=(1,2,3))/torch.sum(output_batch**2, axis=(1,2,3)))).item()

            test_err /= len(testing_set)
            rel_test_err /= len(testing_set)

            historic_rel_test.append(rel_test_err)
            historic_loss_test.append(test_err)

            if not validation_set is None:

                model.eval()
                val_err = 0.
                rel_val_err = 0.

                for _, (input_batch, output_batch) in enumerate(validation_set):

                    loss_f = 0.
                    output_pred_batch = model(input_batch)
                    loss_f = l(output_pred_batch, output_batch).float()

                    val_err += loss_f.item()
                    rel_val_err += torch.mean(torch.sqrt(torch.sum((output_pred_batch-output_batch)**2, axis=(1,2,3))/torch.sum(output_batch**2, axis=(1,2,3)))).item()

                val_err /= len(validation_set)
                rel_val_err /= len(validation_set)

                historic_rel_val.append(rel_val_err)
                historic_loss_val.append(val_err)

                
        avg_time = (epoch*avg_time + time.time() - start_time_epoch)/(epoch+1)
        current_training_time = time.time() - start_time_training

        print("Epoch {}/{}: training loss: {:.2e} --- testing loss: {:.2e} --- rel. train error: {:.2f} --- rel.test error: {:.2f} --- rel.val error: {:.2f} --- average time/epoch: {:.2f} --- training_time: {:.2f}s".format(epoch+1, epochs, train_err, test_err, rel_train_err, rel_test_err, (0. if validation_set is None else rel_val_err), avg_time, current_training_time))

    dict_loss = {"rel_train": historic_rel_train, 
                 "loss_train": historic_loss_train,
                 "rel_test": historic_rel_test, 
                 "loss_test": historic_loss_test,
                 "rel_val": historic_rel_val, 
                 "loss_val": historic_loss_val,
                 "learning_rate": historic_lr}
    
    return(dict_loss)









def train_fourier_1D(model, training_set, testing_set, epochs, learning_rate, validation_set=None, wd=1e-5, l=torch.nn.MSELoss()):
    """ This function train a model. It performs 'epochs' epochs, the optimizer is Adam('learning_rate') and 
    there is a learning rate optimizer"""
    
    N_batch = len(training_set)

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=wd,lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, epochs=epochs, steps_per_epoch=N_batch)
   
    start_time_training = time.time()
    avg_time = 0.
    current_training_time = 0.

    historic_rel_train = []
    historic_loss_train = []

    historic_rel_test = []
    historic_loss_test = []

    historic_rel_val = []
    historic_loss_val = []

    historic_lr = []

    for epoch in range(epochs):

        start_time_epoch = time.time()
        model.train()
        train_err = 0.
        rel_train_err = 0.

        for i_t, (input_batch, output_batch) in enumerate(training_set):

            optimizer.zero_grad()

            loss_f = 0.
            output_pred_batch = model(input_batch)
            loss_f = l(output_pred_batch, output_batch).float()

            train_err += loss_f.item()
            rel_train_err += torch.mean(torch.sqrt(torch.sum((output_pred_batch-output_batch)**2, axis=(1))/torch.sum(output_batch**2, axis=(1)))).item()

            # Compute gradients
            loss_f.backward()

            # Update the parameters
            optimizer.step()
            scheduler.step()


        print("Boucle faite")
        train_err /= len(training_set)
        rel_train_err /= len(training_set)

        print("Erreur calculée")
        historic_lr.append(optimizer.param_groups[0]['lr']) 
        historic_rel_train.append(rel_train_err)
        historic_loss_train.append(train_err)
        print("loss ajoutée")

        # Compute test loss (it follows the same steps as above)
        with torch.no_grad():

            model.eval()
            test_err = 0.
            rel_test_err = 0.

            for i_t, (input_batch, output_batch) in enumerate(testing_set):

                loss_f = 0.
                output_pred_batch = model(input_batch)
                loss_f = l(output_pred_batch, output_batch).float()

                test_err += loss_f.item()
                rel_test_err += torch.mean(torch.sqrt(torch.sum((output_pred_batch-output_batch)**2, axis=(1))/torch.sum(output_batch**2, axis=(1)))).item()

            test_err /= len(testing_set)
            rel_test_err /= len(testing_set)

            historic_rel_test.append(rel_test_err)
            historic_loss_test.append(test_err)

            if not validation_set is None:

                model.eval()
                val_err = 0.
                rel_val_err = 0.

                for _, (input_batch, output_batch) in enumerate(validation_set):

                    loss_f = 0.
                    output_pred_batch = model(input_batch)
                    loss_f = l(output_pred_batch, output_batch).float()

                    val_err += loss_f.item()
                    rel_val_err += torch.mean(torch.sqrt(torch.sum((output_pred_batch-output_batch)**2, axis=(1))/torch.sum(output_batch**2, axis=(1)))).item()

                val_err /= len(validation_set)
                rel_val_err /= len(validation_set)

                historic_rel_val.append(rel_val_err)
                historic_loss_val.append(val_err)

                
        avg_time = (epoch*avg_time + time.time() - start_time_epoch)/(epoch+1)
        current_training_time = time.time() - start_time_training

        print("Epoch {}/{}: training loss: {:.2e} --- testing loss: {:.2e} --- rel. train error: {:.2f} --- rel.test error: {:.2f} --- rel.val error: {:.2f} --- average time/epoch: {:.2f} --- training_time: {:.2f}s".format(epoch+1, epochs, train_err, test_err, rel_train_err, rel_test_err, (0. if validation_set is None else rel_val_err), avg_time, current_training_time))

    dict_loss = {"rel_train": historic_rel_train, 
                 "loss_train": historic_loss_train,
                 "rel_test": historic_rel_test, 
                 "loss_test": historic_loss_test,
                 "rel_val": historic_rel_val, 
                 "loss_val": historic_loss_val,
                 "learning_rate": historic_lr}
    
    return(dict_loss)