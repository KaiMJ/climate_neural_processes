SFNP_tuned = {'hyper_params':
              {'weight_decay': 0.09636627605010294, 'lr': 0.000168},
              'hyper_model_params': {
                  'hidden_layers': 7, 'z_hidden_layers': 7,
                  'z_hidden_dim': 32, 'z_dim': 64, 'hidden_dim': 64},
              'loss': 2.0881555945212398e-05}
SFAttn_tuned = {'hyper_params':
                {'weight_decay': 0.06818202991034834, 'lr': 0.000139},
                'hyper_model_params': 
                {'num_heads': 16, 'attention_layers': 12, 'n_embd': 128, 'hidden_dim': 160, 'dropout': 0.1748127815197366}, 
                'loss': 1.549148262311668e-05}
MFNP_tuned = {'hyper_params': {'weight_decay': 0.0299554994861006, 'lr': 9.9e-05},
              'hyper_model_params': {'hidden_layers': 5, 'z_hidden_layers': 5, 'z_hidden_dim': 64, 'z_dim': 96, 'hidden_dim': 128},
              'loss': 3.570982407000392e-05}
Transformer_tuned = {'hyper_params': 
                     {'weight_decay': 0.04227092357481125, 'lr': 7.7e-05}, 
                     'hyper_model_params': 
                     {'num_heads': 4, 'attention_layers': 12, 'n_embd': 48, 'dropout': 0.0057435876268728155}, 
                     'loss': 1.9375793477618022e-05}
