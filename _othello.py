# othello.py
from torch.nn import Module
import torch
import numpy as np
from torch import nn
from numba import njit
import numpy as np
import random
import numba
import multiprocessing as mp
###
# Channel attention 
###
class ChannelAttention(nn.Module):
    def __init__(self, channels : int = 64 , scale_factor : int = 8) -> None:
        super(ChannelAttention,self).__init__()
        self.scale_factor = scale_factor

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // self.scale_factor, kernel_size = 1,padding = 0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels // self.scale_factor, channels, kernel_size = 1,padding = 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return out
###
# Spatial Attention
###
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size : int = 3) -> None:
        super(SpatialAttention,self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size = kernel_size, padding = kernel_size // 2, bias = False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim = 1, keepdim = True)
        max_out, _ = torch.max(x, dim = 1, keepdim = True)
        x = torch.cat([avg_out, max_out], dim = 1)
        x = self.conv(x)
        return self.sigmoid(x)
###
# Convolutional Block Attention Module
###
class CBAM(nn.Module):
    def __init__(self, channels : int = 64) -> None:
        super(CBAM, self).__init__()
        # conv block attention module
        self.channel_attention = ChannelAttention(channels = channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
###
# residual block
###
class ResidualBlock(nn.Module):
    def __init__(self, channels : int = 64, groups : int = 4, stride : int =1, downsample = None) -> None:
        super(ResidualBlock, self).__init__()
        group_width = channels * groups
        self.conv = nn.Sequential(
            #########################
            nn.Conv2d(channels, group_width, kernel_size=3, stride=1, padding=1, bias=False), # large kernel inspierd by ConvNext
            nn.GroupNorm(num_groups = groups, num_channels = group_width, affine=True),
            ##########################
            CBAM(channels = group_width), # apply attention
            ##########################
            nn.Conv2d(group_width, group_width, kernel_size=1, stride=stride, padding=0, groups=groups),
            nn.LeakyReLU(inplace=True),
            #########################
            nn.Conv2d(group_width, channels, kernel_size=1, stride=1, padding=0),
        )

        self.shortcut = downsample if downsample else nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv(x)
        x += shortcut
        return x
###
#conv block
###
class Block(nn.Module):

        def __init__(self, downsample = None,channels = 32,double=False):
            super(Block,self).__init__()
            self.double = double
            self.conv_1 = nn.Sequential(
                nn.Conv2d(in_channels = channels,out_channels=channels, kernel_size=(3,3),
                                    padding=1,bias=False),
                nn.GroupNorm(num_groups=1,num_channels=channels,affine=True),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels = channels,out_channels=channels, kernel_size=(3,3),
                        padding=1,bias=False),
                nn.GroupNorm(num_groups=1,num_channels=channels,affine=True),
                nn.LeakyReLU(),
            ) if double else None

            self.conv_2 = nn.Sequential(
                nn.Conv2d(in_channels = channels,out_channels=channels, kernel_size=(3,3),
                                    padding=1,bias=False),
                nn.GroupNorm(num_groups=1,num_channels=channels,affine=True),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels = channels,out_channels=channels, kernel_size=(3,3),
                        padding=1,bias=False),
                nn.GroupNorm(num_groups=1,num_channels=channels,affine=True),
                nn.LeakyReLU(),
            )

            self.downsample = downsample if downsample else nn.Identity()

        def forward(self,x):

                x = x + self.conv_1(x) if self.double else x
                return self.downsample(torch.cat((x,self.conv_2(x)),axis = 1))
###      
#value network
###
class value_Net(nn.Module):

    def __init__(self):
        super(value_Net,self).__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels = 3,out_channels=64, kernel_size=(3,3),
                    padding=1,bias=False),
            nn.GroupNorm(num_groups=4,num_channels=64,affine=True),
            nn.LeakyReLU()
            )
        self.res_blocks = nn.Sequential(
            ResidualBlock(channels = 64),
            ResidualBlock(channels = 64),
            ResidualBlock(channels = 64),
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(in_features = 1024,out_features=1),
            nn.Tanh(),
        )
    def forward(self,x):
        return self.fc(self.res_blocks(self.conv_in(x)))
###
#Othello agent
###
class OthelloRL(Module):

    def __init__(self,internal_channels=64,res_depth=8,name="A",device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),return_val = False):
        super(OthelloRL, self).__init__()

        # Input convolutional layer
        self.InConv = nn.Sequential(
            # input is board and opponent's pieces under attack
            nn.Conv2d(in_channels= 2, out_channels=internal_channels,kernel_size=(3,3), padding=1, bias=False),
            nn.GroupNorm(num_groups = 4, num_channels = internal_channels, affine=True),
            nn.LeakyReLU(inplace=True)
        )

        # ResidualBlock block
        self.ResidualBlock = nn.Sequential(
                *[ResidualBlock(channels = internal_channels) for _ in range(res_depth)]
        )

        # Output convolutional layer
        self.OutConv = nn.Sequential(
            nn.Conv2d(in_channels=internal_channels, out_channels=1,
                      kernel_size=(1,1), bias=True)
        )

        self.value_net = value_Net() # value network

        self.internal_channels = internal_channels
        self.res_depth = res_depth
        self.name = name
        self.return_val = return_val
        self.device = device

        self.softmax = nn.Softmax(dim=1)
    ##############################
    def forward(self, input , mask):
        x = input
        x = self.InConv(x)
        x = self.ResidualBlock(x)
        x = self.OutConv(x)
        # Apply masking to prevent choosing already chosen positions
        x = x*mask + (1 - mask)*-1e32 # Set chosen positions to a large negative value
        x = self.softmax(x.view(x.size(0), -1)).view(x.size())

        if self.return_val :

            value = self.value_net(torch.cat((input,x),axis = 1))
            return x,value
        
        return x
    ###############################
    def save(self,path):
      #function returns true when object is saved, false when object isn't saved
      try :
        torch.save(self,path)
        return True
      except Exception as e :
        print("Model Saving Failed :",e)
        return False

    def load(self,path):

      #load model from path otherwise return false and print error message
      try:
        loaded_model = torch.load(path,map_location=self.device)
        self.__dict__.update(loaded_model.__dict__)  # Update current instance with loaded model's attributes
        return True

      except Exception as e :
        print(f"Model {self.name} wasn't found :",e)
        return False
###
# minimax engine
###
class miniMax(nn.Module):
    def __init__(self, depth:int = 5, name:str = "Engine",return_val:bool = False)-> None :
        super(miniMax, self).__init__()

        self.depth = depth
        self.name = name
        self.return_val = return_val

    def forward(self, input, mask):

        board = input[0][0].cpu().numpy()

        transposition_table = numba.typed.Dict.empty(
            key_type=numba.types.int64,
            value_type=numba.types.Tuple((numba.types.float64, numba.types.Tuple((numba.types.int64, numba.types.int64)), numba.types.boolean)),
        )
        
        value, (x, y) = minimax(board,transposition_table = transposition_table,depth=self.depth )

        result = torch.zeros((1, 1, board.shape[0], board.shape[1]))
        result[0][0][x][y] = 1.0

        return (result, value/(board.shape[0]*board.shape[1])) if self.return_val else result
###
# hash
###
@numba.njit
def board_hash(board) -> int:
    # Compute the hash value of the board array
    # You can use any hash function that takes a 2D array of floats as input
    # and returns a hash value of type `int64`
    hash_value = 0
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            hash_value = (hash_value * 991 + int(board[i, j])) & 0xFFFFFFFFFFFFFFFF
    return hash_value
###
# minimax function
###
@njit
def minimax(board,transposition_table,depth:int=3,maximizing_player:bool=True,alpha=-np.inf,beta=np.inf):
    
    if depth == 0 or np.all(board != 0):
        return evaluate(board),(0,0)
    
    key = board_hash(board) 

    if key in transposition_table:
        score,best_move,maximizing_player_prev = transposition_table.get(key)
        if maximizing_player_prev == maximizing_player:
            return score,best_move

    best_move = None

    if maximizing_player:
        max_eval = -np.inf
        legal_moves, _, move_board = get_valid_moves(board)
        if np.any(legal_moves):
            for move, conversion_board in move_board.items():
                board_clone = np.copy(board)
                board_clone[move[0]][move[1]] = 1
                board_clone = np.where(conversion_board,1.0,board_clone)
                eval_score, _ = minimax(board_clone,transposition_table,depth - 1,maximizing_player=False,alpha=alpha,beta=beta)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
        else:
            return evaluate(board),(0,0)

        transposition_table.update({key :(max_eval,best_move,maximizing_player)})

        return max_eval,best_move
    else:
        min_eval = np.inf
        legal_moves, _, move_board = get_valid_moves(-1.0*board)
        if np.any(legal_moves):
            for move, conversion_board in move_board.items():
                board_clone = np.copy(board)
                board_clone[move[0]][move[1]] = -1.0
                board_clone = np.where(conversion_board,-1.0,board_clone)
                eval_score, _ = minimax(board_clone,transposition_table,depth - 1,maximizing_player=True,alpha=alpha,beta=beta)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
        else:
            return evaluate(board),(0,0)
        
        transposition_table.update({key :(min_eval,best_move,maximizing_player)})

        return min_eval,best_move
###
# boltzmann exploration
###
def decide(probabilities):
    probabilities = probabilities / probabilities.sum()
    indices = np.random.choice(probabilities.size, size=1, p=probabilities.ravel())
    i, j = indices[0] // probabilities.shape[1], indices[0] % probabilities.shape[1]
    return i, j
###
# evaluation function
###
@njit
def evaluate(board) -> float:
    return np.sum(board).item()
###
# get valid move from the board
###
@njit
def get_valid_moves(board):
    size = board.shape[0]
    legal_moves = np.zeros_like(board,dtype=np.bool_)
    possible_conv = np.zeros_like(board, dtype=np.bool_)

    move_board = {
        # save each move and the resulting board
    }
    for row_index in range(size):
        for col_index in range(size):
            if board[row_index, col_index] == 0.0:
                # Check in all eight directions
                row_min, row_max = max(row_index-1, 0), min(row_index+2, size)
                col_min, col_max = max(col_index-1, 0), min(col_index+2, size)
                dir = [(board[i, j] == -1.0, (i - row_index, j - col_index)) for i in range(row_min, row_max) for j in range(col_min, col_max)]

                # Check if any direction has a black piece
                shortlisted = False
                for indicator, _ in dir:
                    if indicator:
                        shortlisted = True
                        break

                if shortlisted:
                    # Use dir to check if a conversion will happen in this direction
                    conversion = np.zeros(board.shape, dtype=np.bool_)
                    sup_conv = np.zeros(board.shape, dtype=np.bool_) # converts from a single move
                    for i in range(len(dir)):
                        indicator, (dy, dx) = dir[i]
                        if indicator:
                            ni, nj = row_index + dy, col_index + dx
                            temp = np.zeros(board.shape, dtype=np.bool_)
                            while 0 <= ni < size and 0 <= nj < size and board[ni, nj] == -1.0:
                                temp[ni, nj] = True
                                ni, nj = ni + dy, nj + dx

                            if 0 <= ni < size and 0 <= nj < size and board[ni, nj] == 1.0:
                                conversion = np.logical_or(conversion,temp)
                                sup_conv = np.logical_or(sup_conv,temp)
                      # save for use later in creating moves
                    # If conversion is possible, update legal moves and possible_conv
                    if np.any(conversion):
                        legal_moves[row_index, col_index] = True
                        possible_conv = np.logical_or(possible_conv, sup_conv)
                        move_board[(row_index, col_index)] = sup_conv


    return legal_moves, possible_conv,move_board
###
# Othello Board class
###
class OthelloBoard:

    def __init__(self, size : int =8) -> None :

        self.size = size
        self.gamma = np.exp(np.log(0.01)/(size**2 - 4)) # dynamic gamma accourding to the board size
        self.board = np.zeros((self.size, self.size), dtype= np.short)
        ##########################################################
        self.board[self.size//2 - 1][self.size//2 - 1] = 1.0
        self.board[self.size//2][self.size//2] = 1.0
        self.board[self.size//2 - 1][self.size//2] = -1.0
        self.board[self.size//2][self.size//2 - 1] = -1.0
        ########################################################
        self.player = True # True for white and False for black
        ########################################################
        self.white_history = []
        self.black_history = []
        ########################################################
        self.white_mask = []
        self.black_mask = []
        ########################################################
        self.white_actions = []
        self.black_actions = []
        ########################################################
        self.white_return = []
        self.black_return = []
        ########################################################
        self.white_return_scale = []
        self.black_return_scale = []
        #######################################################
        self.last_player_moved = True  # save if the last player was able to move

    def resetBoard(self) -> None:
        # reset the board to the initial state
        self.board = np.zeros(size=(self.size, self.size), dtype= np.short)
        ##########################################################
        self.board[self.size//2 - 1][self.size//2 - 1] = 1.0
        self.board[self.size//2][self.size//2] = 1.0
        self.board[self.size//2 - 1][self.size//2] = -1.0
        self.board[self.size//2][self.size//2 - 1] = -1.0
        ########################################################
        self.player = True # True for white and False for black
        #######################################################
        self.last_player_moved = True  # save if the last player was able to move

    def resetMem(self) -> None:
        # remove the state of the system except the board and the player
        ########################################################
        self.white_history = []
        self.black_history = []
        ########################################################
        self.white_mask = []
        self.black_mask = []
        ########################################################
        self.white_actions = []
        self.black_actions = []
        ########################################################
        self.white_return = []
        self.black_return = []
        ########################################################
        self.white_return_scale = []
        self.black_return_scale = []

    def getBoard(self) -> None:
        # return the board as numpy array
        # return all possible moves
        # return opponents pieces under attack
        # return the mask
        mask, under_attack, moves = get_valid_moves(self.board)
        element = (

            np.expand_dims(np.concatenate((np.expand_dims(self.board, axis=0), np.expand_dims(under_attack, axis=0)), axis=0), axis=0).astype(np.float32),
            np.expand_dims(np.expand_dims(mask, axis=0), axis=0).astype(np.float32)

        )

        return element, moves


    def switchPlayer(self) -> bool :
        self.player = not self.player
        self.board = np.rot90(self.board, k=2) * -1
        #self.board.neg_().rot90(2, dims=(0, 1))
        return self.player

    def drawBoard(self, background_color = np.array([238,238,210]),boarder_color = np.array([92, 64, 51]),quality_factor = 128, boarder_thickness = 2, checker_size = "m"):
        # every block of the board will be quality_factor x quality_factor pixels
        board_shape = self.board.shape # getting the shape of the board
        image_len = self.size*quality_factor # the w & h of the image
        image = np.ones(shape=(image_len,image_len,1))*background_color # creating a plank image with the background color
        mask,_,_ = get_valid_moves(self.board)
        # draw boxes
        for i in range(0,image_len):
          if i % quality_factor < boarder_thickness or quality_factor - i % quality_factor < boarder_thickness:
            # drawing horizontal boarders
            image[i, :] = boarder_color
            #drawing vertical boarders
            image[:, i, :] = boarder_color
        ####################################################
        image = image/255.0 # normalize the image 0 -> 1
        checker = np.ones(shape=(quality_factor,quality_factor,3)) # generating a checker
        #draw the actuall checker
        x = (7 - (1 if checker_size == "m" else 2 if checker_size == "l" else 0) )*(quality_factor//4)**2
        for i in range(0,quality_factor):
          for j in range(0,quality_factor):
            ############################################################
            if i**2 + j**2 - quality_factor*i - quality_factor*j + x > 0:
              checker[i,j] = np.array([0,0,0])

        map_rows, map_cols, _ = image.shape
        # butting checkers on the board
        for raw_index, raw in enumerate(self.board):
            for cell_index, cell in enumerate(raw):
                raw_start, raw_end = raw_index * quality_factor, raw_index * quality_factor + quality_factor
                cell_start, cell_end = cell_index * quality_factor, cell_index * quality_factor + quality_factor

                if int(cell) == 1:
                    # Update the image for white pieces
                    image[raw_start:raw_end, cell_start:cell_end] = (
                        image[raw_start:raw_end, cell_start:cell_end] * (1 - checker) + checker)

                if int(cell) == -1:
                    # Update the image for black pieces
                    image[raw_start:raw_end, cell_start:cell_end] = (
                        image[raw_start:raw_end, cell_start:cell_end] * (1 - checker))

                if mask[raw_index][cell_index] == 1:
                    # Update the image for possible moves (make them darker)
                    image[raw_start:raw_end, cell_start:cell_end] = (
                        image[raw_start:raw_end, cell_start:cell_end] * 0.85)

        return image

    def modelMove(self, OthelloRL,switching:bool=True,epsilon:float=0.0,boltzmann:bool = True,device = None)->bool:
        # let the model make move
        with torch.no_grad():
            (x,mask),all_moves = self.getBoard()

            if np.random.rand() >= epsilon :
                if device:
                        output_tensor = OthelloRL(torch.from_numpy(x).to(device)
                                                  ,torch.from_numpy(mask).to(device)).cpu().numpy()
                else :
                        output_tensor = OthelloRL(torch.from_numpy(x)
                                                  ,torch.from_numpy(mask)).numpy()

                if len(all_moves) > 0 : # check if any move is available
                    if boltzmann :
                        # boltzmann exploration
                        I,J = decide(probabilities = np.squeeze(output_tensor, axis=0))
                    else :
                        # greedy exploration
                        max_indices = np.argmax(output_tensor.reshape(-1, output_tensor.shape[-2] * output_tensor.shape[-1]), axis=1)
                        I, J = np.divmod(max_indices, output_tensor.shape[-1])
                        I,J = I[0],J[0]

                    board_change = all_moves[(I,J)]
                else :
                    I,J = None,None
                    board_change = None # no move
                    # game ends or player has no move
            else :
                # random exploration
                if len(all_moves) > 0 :
                    I,J = random.choice(list(all_moves.keys()))
                    board_change = all_moves[(I,J)]
                else :
                    I,J = None,None
                    board_change = None # no move
                    # game ends or player has no move

        game_state = self.makeMove(helpers = (x,mask), x=I, y=J, board_change = board_change,
                                   switching=switching)
        # return if the game ended
        return game_state


    def humanMove(self,x : int,y : int, switching:bool=True) -> bool:
            # let the model make move
            (X,mask),all_moves = self.getBoard()
            if len(all_moves) == 0 :
                I,J = None,None
                board_change = None
            elif (x,y) not in list(all_moves) :
                
                raise Exception(f"Invalid move : {x,y}")
            else :
                board_change = all_moves[(x,y)]
                I,J = x,y

            game_state = self.makeMove(helpers = (X,mask), x=I, y=J, board_change = board_change,
                                      switching=switching)

            return game_state


    def makeMove(self,helpers, x:int=0, y:int=0,board_change= None,switching:bool = True)->bool:
        # given x & y this function is responseble of making moves on the board and switch player for the other turn


        if board_change is not None :
            if (self.board[x,y] != 0.0):
                print(f"{x,y} is already taken.")

            self.last_player_moved = True
            infrence_input,mask = helpers
            # check if the possition is inside the board
            self.saveBoard(x,y,infrence_input = infrence_input,mask = mask)
            # take the possition
            self.board[x, y] = 1.0

            self.board[board_change] = 1.0
        else :
            if not self.last_player_moved :
                # if last player also wasn't able to move
                self.switchPlayer()
                return False
            else :
                self.last_player_moved = False

        # switching player for the next turn
        if switching:
            self.switchPlayer()

        return False if (self.board != 0).all() else True



    def saveBoard(self, x, y, infrence_input, mask):
        # Save the board for training later
        board = np.zeros((1, 1, self.size, self.size))
        board[0, 0, x, y] = 1

        scalar_array = self.gamma ** np.sum(self.board == 0)

        ############################
        if self.player:
            self.white_history.append(np.squeeze(infrence_input))
            self.white_mask.append(np.squeeze(mask,axis = 0))
            self.white_actions.append(np.squeeze(board,axis = 0))
            self.white_return_scale.append(scalar_array)
        else:
            self.black_history.append(np.squeeze(infrence_input))
            self.black_mask.append(np.squeeze(mask,axis = 0))
            self.black_actions.append(np.squeeze(board,axis = 0))
            self.black_return_scale.append(scalar_array)



    def selfPlay(self,OthelloRL,start_explore = (0,32), switching=True, epsilon=0.0,boltzmann = True,device = None): # self play one game
            start,finish = start_explore # play a random number of moves
            start_explore_depth = random.randint(start, min(finish,32)) # upper bound of 32
            ##################################################

            for _ in range(0,start_explore_depth):
                self.modelMove(OthelloRL, switching=True, epsilon=1.0,boltzmann =False,device = device)

            self.resetMem()

            while self.modelMove(OthelloRL,switching= switching, epsilon= epsilon ,boltzmann = boltzmann,device = device) :
                pass# play to the end of the game

            delta = evaluate(board = self.board)

            winner = self.player if delta > 0 else not self.player if delta < 0 else -1
            # no actions
            if len(self.white_actions) == 0 and len(self.black_actions) == 0 :
                return winner, None

            data = []
            for i in range(len(self.white_actions)):

                white_return = 0.0 if winner == -1 else self.white_return_scale[i].astype(np.float32) if winner else -1*self.white_return_scale[i].astype(np.float32)

                data.append(((self.white_history[i].astype(np.float32),self.white_mask[i].astype(np.float32)), self.white_actions[i].astype(np.float32), white_return ))

            for i in range(len(self.black_actions)):

                black_return = 0.0 if winner == -1 else -1**self.black_return_scale[i].astype(np.float32) if winner  else self.black_return_scale[i].astype(np.float32)

                data.append(((self.black_history[i].astype(np.float32),self.black_mask[i].astype(np.float32)),self.black_actions[i].astype(np.float32),black_return))

            
            return winner, data


    def modelVsModel(self,OthelloRL_1,OthelloRL_2,start_explore = (0,32),Epsilon_1 = 0.0,Epsilon_2 = 0.0, boltzmann = False):
          # this function returns ture,false if the first model won
          # draw => false,false
          # lost => false,true
          # true,true is invalid
          start,finish = start_explore # play a random number of moves
          start_explore_depth = random.randint(start, min(finish,62)) # upper bound of 32
            ##################################################

          for _ in range(0,start_explore_depth):
              self.modelMove(OthelloRL_1, switching=True, epsilon=1.0,boltzmann =False)

          singal = True # true when the game ends

          while True :

              singal = self.modelMove(OthelloRL=OthelloRL_1,epsilon = Epsilon_1,boltzmann = boltzmann)
              if not singal :
                delta = evaluate(board = self.board)
                return np.array([delta < 0, delta > 0])

              singal = self.modelMove(OthelloRL=OthelloRL_2,epsilon = Epsilon_2,boltzmann = boltzmann)
              if not singal :
                delta = evaluate(board = self.board)
                return np.array([delta > 0, delta < 0])
###
# play a number of games for single worker
###
def asyncSelfPlay(model,device,Epsilon : float = 0.0,boltzmann:bool = False,game_per_worker:int = 10,board_size:int = 8) -> list:
    data = []
    for game in range(game_per_worker):
        # Define the linear model
        Board = OthelloBoard(size=board_size)  # generate a random board with dynamic size
        ##################################################
        _,_data = Board.selfPlay(model,epsilon=Epsilon,device = device,boltzmann = boltzmann)  # self-playing
        ##################################################
        data = data + _data if _data else data

    return data
###
# save played games to the queue
###
def process_func(model,device, results_queue,Epsilon,boltzmann,game_per_worker,board_size) -> None:

    result = asyncSelfPlay(model,device,Epsilon = Epsilon,boltzmann = boltzmann,game_per_worker = game_per_worker,board_size = board_size)
    results_queue.append(result)
###
# play games in multible workers
###
def parallelSelfPlay(model,device,workers:int=4,Epsilon:float = 0.0,boltzmann:bool = False,game_per_worker:int=10,board_size:int =8) -> list:

    results_queue = mp.Manager().list()

    model.to(device)
    model.eval()
    model.share_memory()

    processes = []
    
    for rank in range(workers):
        p = mp.Process(target=process_func, args=(model,device,results_queue,Epsilon,boltzmann,game_per_worker, board_size), name=f'self playing worker -{rank}') 
        p.start()
        processes.append(p) 
        print(f'Started : {p.name}')

    for p in processes: 
        p.join() 
        print(f'Finished : {p.name}') 

    result =[]
    for r_q in results_queue:
        result = result + r_q

    return result
###
# proximal policy loss
###
class PPO_Loss(nn.Module):
    def __init__(self,clip_param:float=0.25,value_coefficient:float=0.5,entropy_coefficient:float=0.01,echo:bool=False):
        super(PPO_Loss, self).__init__()
        self.clip_param = clip_param
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.mse_loss = torch.nn.MSELoss()
        self.echo = echo

    def compute_entropy_loss(self,policy):
        epsilon = 1e-8  # Small epsilon value to prevent taking the logarithm of zero
        policy = torch.clamp(policy, epsilon, 1.0 - epsilon)  # Clamp probabilities to avoid extreme values
        policy = policy.view(policy.size(0), -1)
        entropy_loss = -torch.mean(torch.sum(policy * torch.log(policy), dim=1))
        return entropy_loss

    def forward(self, policy_old, policy_new, actions, value_head_output, returns):
        # Compute advantages
        ret = returns.unsqueeze(-1).float()
        advantages = ret - value_head_output.detach()
        # Compute probability ratios and clip them
        prob_act_old = (policy_old.detach() * actions).sum(dim=(2, 3), keepdim=True).squeeze(-1).squeeze(-1) + 1e-9
        prob_act_new = (policy_new * actions).sum(dim=(2, 3), keepdim=True).squeeze(-1).squeeze(-1)
        ratio = prob_act_new / prob_act_old
        clipped_ratios = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        # Compute policy loss
        policy_loss = -torch.where(torch.abs(advantages*ratio)>torch.abs(advantages*clipped_ratios),advantages*clipped_ratios,advantages*ratio)
        policy_loss = policy_loss.mean()
        # Compute value loss
        value_loss = self.mse_loss(value_head_output, ret)
        # Compute entropy loss
        entropy_loss = self.compute_entropy_loss(policy_new)
        # Total loss
        #print(f"policy_loss : {policy_loss}, value loss : {value_loss}, entropy_loss :{entropy_loss}.")
        total_loss = policy_loss + self.value_coefficient * value_loss - self.entropy_coefficient * entropy_loss

        if self.echo :
            return total_loss, policy_loss, value_loss, entropy_loss, torch.mean(advantages.detach())
        else :
            return total_loss
##################################################################################################################################################################

  #####   #######  #      #  #####  #       #        ##### 
 #     #     #     #      #  #      #       #       #     #  
 #     #     #     #      #  #      #       #       #     #  
 #     #     #     ########  #####  #       #       #     #
 #     #     #     #      #  #      #       #       #     #  
 #     #     #     #      #  #      #       #       #     #  
  #####      #     #      #  #####  ######  ######   ##### 

#################################################################################################################################################################
