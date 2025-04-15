function r = Gen_recipient(lastname, firstname)

% Gen_recipient: Retrieve recipient's MAP numbers and dates from the Cochlear database.
% Archived recipients will be omitted.
% Combinations of last and first names in the database must be unique.
%
% r = Gen_recipient(lastname, firstname)
%
% Inputs:
% lastname:             Recipient's last name
% firstname:            Recipient's first name
%                       
% Outputs:
% r:                    Recipient struct with additional fields:
% r.map_numbers:        Recipient's MAP numbers 
% r.map_dates:          Recipient's MAP creation dates 
%
% See also: Get_recipient_list, Get_data.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Copyright: Cochlear Ltd
%      $Change: 86418 $
%    $Revision: #1 $
%    $DateTime: 2008/03/04 14:27:13 $
%      Authors: Herbert Mauch
%               credits to Michael Büchler (USZ Zürich)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

r = [];
r.lastname = lastname;
r.firstname = firstname;

%% Create SQL query and retreive data
querytext = sprintf(['SELECT     MAP.MAPNumber, MAP.Date_Creation' ...
                     ' FROM         dbo.MAP INNER JOIN' ...
                     ' dbo.Implant ON dbo.MAP.FK_GUID_Implant = dbo.Implant.GUID_Implant ' ...
                     ' INNER JOIN dbo.Recipient ON dbo.Implant.FK_GUID_Recipient = ' ... 
                     'dbo.Recipient.GUID_Recipient WHERE (Recipient.T_RecordStatus = 0)' ...
                     ' AND (MAP.T_RecordStatus = 0) AND ' ...
                     ' (Recipient.Name_Last = ''' r.lastname ''') AND ' ...
                     '(Recipient.Name_First = ''' r.firstname ''') ORDER BY MAP.MAPNumber']);

data = Get_data(querytext);

r.map_numbers = data.MAPNumber;
r.map_dates   = data.Date_Creation;