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
  return image.scale(frame, 60, 50)
end

if table.getn(arg) > 1 then
  local input=arg[1]
  local output=arg[2]
  local action=torch.load(input)
  print(action:size())
  decompose_action(output,action)
end
