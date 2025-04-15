function data = Get_data(querytext)

% Get_data: Get specified data query from the Cochlear database.
% Get_data establishes the connection to the Cochlear Database,
% executes the SQL query passed, returns the data as a struct and 
% closes the database connection.
% This function requires the Matlab Database Toolbox to be installed.
% It requires an ODBC database connection to the respective Cochlear Database
% to be setup via Control Panel/Administrative Tools/Data Source (ODBC).
% See Matlab help for further information.
%
% data = Get_data(querytext)
%
% Inputs:
% querytext:            SQL query to be executed
%
% Outputs:
% data:                 data struct containing the query data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Copyright: Cochlear Ltd
%      $Change: 86418 $
%    $Revision: #1 $
%    $DateTime: 2008/03/04 14:27:13 $
%      Authors: Herbert Mauch
%               credits to Michael Büchler (USZ Zürich)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Connect to database
Databasename    = 'Cochlear_Database';                              % Name of ODBC Connection
Username        = '';                   %'sa';                      % Username & Password can be left blank
Password        = '';                   %'littlefoot is coming';    % On some installations the strong user 'sa' has to be used
    
%logintimeout(5);                                                    % conncetion timeout in seconds
connection = database(Databasename, Username, Password);            % connect to Cochlear Database
if ~isempty(connection.Message)                                     % no message is a good message
    disp('Database Connection Failed!');
    disp(connection.Message);
    return;
end
setdbprefs('DataReturnFormat','structure');

%% Execute the SQL query
curs=exec(connection, querytext);                                   % execute SQL query
curs=fetch(curs);                                                   % retreive data
if ~isstruct(curs.Data)
    error(['No data retrieved. Message: ' curs.Message]);
else
    data = curs.Data;
end

%% Close database connection
close(curs);
close(connection);