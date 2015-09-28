require 'torch'
require 'math'
require 'image'

function decompose_action(output,action)
  local nframes=action:size()[1]
  for i=1,nframes do
    local filename=output .. "_" .. tonumber(i) ..".jpg"
    local frame=action[i]
    local scaled_frame=scale_frame(frame)
    image.save(filename, scaled_frame)
  end
end

function scale_frame(frame)
  frame=remove_nonzero(frame)
  return image.scale(frame, 60, 50)
end

function remove_nonzero(frame)
  local extreme_points=find_extreme_points(frame)
  local min=extreme_points[1]
  local max=extreme_points[2]
  return frame:sub(min[1],max[1],min[2],max[2])
end

function find_extreme_points(frame)
  local size=frame:size()
  local min_x=nil
  local min_y=nil--size[2]
  local max_x=1
  local max_y=1
  for x_i=1,size[1] do
    for y_i=max_y,size[2] do
      if not (frame[x_i][y_i]==0) then
         max_x=x_i
         max_y=y_i
         if not min_x then
           min_x=x_i
         end 
      end
    end
  end
  min_y=max_y
  for x_i=max_x,size[1] do
    for y_i=1,size[2] do
      if not (frame[x_i][y_i]==0) then
         max_x=x_i
         if y_i<min_y then
           min_y=y_i
         end
         break 
      end
    end
  end
  return pack_extreme(min_x,min_y,max_x,max_y)
end

function pack_extreme(min_x,min_y,max_x,max_y)
  return torch.Tensor({{min_x,min_y},{max_x,max_y}})
end

if table.getn(arg) > 1 then
  local input=arg[1]
  local output=arg[2]
  local action=torch.load(input)
  print(action:size())
  decompose_action(output,action)
end
