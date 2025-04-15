function list = Get_recipient_list(incl_archived)

% Get_recipient_list: Retrieve a list of all recipients from the Cochlear database.
% The list is ordered by last name, first name and date of birth.
% By default archived recipients are omitted.
%
% list = Get_recipient_list(incl_archived)
%
% Optionsl inputs:
% incl_archived:        Logical input to include archived recipients
%
% Outputs:
% list:                 Listing containing all recipients
%                       sorted by last name, first name, date of birth
%
% See also: Get_data.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Copyright: Cochlear Ltd
%      $Change: 86418 $
%    $Revision: #1 $
%    $DateTime: 2008/03/04 14:27:13 $
%      Authors: Michael Büchler (USZ Zürich), Herbert Mauch
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Create SQL query and retreive data
excl = sprintf(['select Name_First, Name_Last, Date_Birth'...
        ' from Recipient where T_RecordStatus = 0'... %exclude archived recipients
        ' order by Name_Last, Name_First, Date_Birth' ]);
incl = sprintf(['select Name_First, Name_Last, Date_Birth'...
        ' from Recipient'...
        ' order by Name_Last, Name_First, Date_Birth' ]);

switch nargin
    case 0
        querytext = excl;
    case 1
    if incl_archived
        querytext = incl;
    else
        querytext = excl;
    end
end

data = Get_data(querytext);

%% convert data into a list
list(:,1) = data.Name_Last;
list(:,2) = data.Name_First;
list(:,3) = data.Date_Birth;